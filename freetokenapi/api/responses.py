"""A local Responses API compatibility layer over Chat Completions."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import Field

from .compat import (
    SSE_HEADERS,
    Backend,
    ChatRun,
    NativeRequest,
    NativeRoute,
    ProtocolError,
    dumps,
    error_from_exception,
    file_part,
    image_part,
    require_object,
    require_string,
    sse,
    text_content,
    tool_definition,
    uid,
    validate_tool_choice,
    validate_tool_names,
)


class ResponsesRequest(NativeRequest):
    model: str = Field(min_length=1)
    input: str | list[dict[str, Any]] = Field(default_factory=list)
    instructions: str | None = None
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = "auto"
    parallel_tool_calls: bool = True
    max_output_tokens: int | None = Field(default=None, gt=0)
    previous_response_id: str | None = None
    store: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    user: str | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    include: list[str] = Field(default_factory=list)
    stream_options: dict[str, Any] | None = None
    background: bool = False
    conversation: Any = None
    prompt: Any = None
    context_management: list[dict[str, Any]] | None = None
    max_tool_calls: int | None = None
    top_logprobs: int | None = None
    service_tier: str | None = None
    truncation: Literal["auto", "disabled"] = "disabled"
    # Existing FreeTokenAPI extensions; not OpenAI service features.
    thinking: bool | None = None
    search: bool | None = None
    session_id: str | None = None


class ResponseStore:
    """Bounded, process-local history. Nothing is written to disk.

    Limits refer to serialized bytes; Python object overhead is additional.
    Multiple uvicorn workers have separate stores, so use a single worker.
    """

    def __init__(self, max_entries: int = 128, max_bytes: int = 32 * 1024 * 1024, ttl: float = 3600):
        self.max_entries, self.max_bytes, self.ttl = max_entries, max_bytes, ttl
        self.entries: OrderedDict[str, tuple[float, int, dict, list[dict]]] = OrderedDict()
        self.size = 0

    def _prune(self):
        now = time.monotonic()
        for key, (expires, _, _, _) in list(self.entries.items()):
            if expires <= now:
                self.delete(key)

    def put(self, response: dict, history: list[dict]) -> bool:
        self._prune()
        size = len(dumps([response, history]).encode("utf-8"))
        if size > self.max_bytes or self.max_entries < 1:
            return False
        self.delete(response["id"])
        while self.entries and (len(self.entries) >= self.max_entries or self.size + size > self.max_bytes):
            self.delete(next(iter(self.entries)))
        self.entries[response["id"]] = (time.monotonic() + self.ttl, size, copy.deepcopy(response), copy.deepcopy(history))
        self.size += size
        return True

    def get(self, key: str) -> tuple[dict, list[dict]]:
        self._prune()
        record = self.entries.get(key)
        if record is None:
            raise ProtocolError(
                "Response not found: it may be unstored, expired, evicted, or from a previous server process. Resend full input history.",
                404,
                "previous_response_id",
                "response_not_found",
            )
        self.entries.move_to_end(key)
        return copy.deepcopy(record[2]), copy.deepcopy(record[3])

    def delete(self, key: str) -> bool:
        record = self.entries.pop(key, None)
        if record is None:
            return False
        self.size -= record[1]
        return True


def _content(value: Any, param: str) -> str | list[dict]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ProtocolError("Message content must be a string or an array of content parts", param=param)
    parts = []
    for index, raw in enumerate(value):
        part = require_object(raw, f"{param}.{index}")
        kind = require_string(part.get("type"), param + ".type")
        if kind in {"input_text", "output_text"}:
            parts.append({"type": "text", "text": require_string(part.get("text"), param, empty=True)})
        elif kind == "refusal":
            parts.append({"type": "text", "text": require_string(part.get("refusal"), param, empty=True)})
        elif kind == "input_image":
            parts.append(image_part(part.get("image_url"), param))
        elif kind == "input_file":
            if part.get("file_id") or part.get("file_url"):
                raise ProtocolError("Use inline file_data and filename; remote files and file IDs are not supported", param=param)
            parts.append(file_part(part.get("file_data"), part.get("filename", "attachment"), param))
        else:
            raise ProtocolError(f"Unsupported Responses content type: {kind}", param=param, code="unsupported_value")
    return text_content(parts)


def input_messages(value: str | list[dict]) -> list[dict]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages: list[dict] = []
    for index, item in enumerate(value):
        param = f"input.{index}"
        kind = require_string(item.get("type", "message"), param + ".type")
        if kind == "message":
            role = require_string(item.get("role"), param + ".role")
            if role not in {"system", "developer", "user", "assistant"}:
                raise ProtocolError("Unsupported message role", param=param + ".role")
            messages.append(
                {"role": "system" if role == "developer" else role, "content": _content(item.get("content"), param + ".content")}
            )
        elif kind in {"function_call", "custom_tool_call"}:
            call_id = require_string(item.get("call_id"), param + ".call_id")
            name = require_string(item.get("name"), param + ".name")
            if kind == "custom_tool_call":
                arguments = dumps({"input": require_string(item.get("input"), param + ".input", empty=True)})
            else:
                arguments = require_string(item.get("arguments"), param + ".arguments", empty=True)
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": ""})
            function = {"name": name, "arguments": arguments}
            namespace = _tool_namespace(item.get("namespace"), param + ".namespace")
            if namespace is not None:
                function["namespace"] = namespace
            messages[-1].setdefault("tool_calls", []).append({"id": call_id, "type": "function", "function": function})
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": require_string(item.get("call_id"), param + ".call_id"),
                    "content": _content(item.get("output"), param + ".output"),
                }
            )
        elif kind == "reasoning":
            # Readable summaries are output annotations, not assistant answers.
            # We cannot decrypt real OpenAI reasoning or fabricate encrypted state.
            if item.get("encrypted_content"):
                raise ProtocolError("Encrypted OpenAI reasoning cannot be replayed by a web-model backend", param=param)
        else:
            raise ProtocolError(f"Unsupported Responses input item: {kind}", param=param, code="unsupported_value")
    return messages


@dataclass(frozen=True)
class ResponseTool:
    name: str
    kind: str
    namespace: str | None = None


def _tool_backend_name(name: str, namespace: str | None) -> str:
    if namespace is None:
        return name
    # Stable, ASCII, <64-character names: no order-dependent numbering or
    # splitting on dots/underscores when restoring an executable tool identity.
    digest = hashlib.sha256(dumps([namespace, name]).encode("utf-8")).hexdigest()[:20]
    group = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)[:12]
    leaf = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:20]
    return f"ns_{group}_{leaf}_{digest}"


def _tool_namespace(value: Any, param: str) -> str | None:
    return None if value is None else require_string(value, param)


def _claim_tool_name(owners: dict[str, tuple[str | None, str]], name: str, namespace: str | None, param: str) -> str:
    alias = _tool_backend_name(name, namespace)
    identity = (namespace, name)
    if alias in owners and owners[alias] != identity:
        raise ProtocolError("Tool identities have conflicting internal names", param=param)
    owners[alias] = identity
    return alias


def _backend_history(history: list[dict], bindings: dict[str, ResponseTool]) -> list[dict]:
    # Cached history stays canonical so changing/reordering the active tools
    # cannot silently redirect a past call to a different namespace.
    messages = copy.deepcopy(history)
    owners = {alias: (tool.namespace, tool.name) for alias, tool in bindings.items()}
    for message in messages:
        for call in message.get("tool_calls", []):
            fn = call["function"]
            namespace = fn.pop("namespace", None)
            fn["name"] = _claim_tool_name(owners, fn["name"], namespace, "input")
    return messages


def _canonical_assistant(message: dict, bindings: dict[str, ResponseTool]) -> dict:
    message = copy.deepcopy(message)
    for call in message.get("tool_calls", []):
        fn = call["function"]
        tool = bindings[fn["name"]]
        fn["name"] = tool.name
        if tool.namespace is not None:
            fn["namespace"] = tool.namespace
    return message


def _tools(req: ResponsesRequest) -> tuple[list[dict], dict[str, ResponseTool]]:
    tools: list[dict] = []
    bindings: dict[str, ResponseTool] = {}
    owners: dict[str, tuple[str | None, str]] = {}
    namespaces: set[str] = set()

    def add_tool(tool: dict, param: str, namespace: str | None = None, namespace_description: str = "") -> None:
        kind = require_string(tool.get("type"), param + ".type")
        if kind not in {"function", "custom"}:
            raise ProtocolError(
                f"Unsupported tool type: {kind}. Only client function/custom tools and their namespace groups are available; "
                "no hosted tools are executed.",
                param=param,
            )
        name = require_string(tool.get("name"), param + ".name")
        alias = _claim_tool_name(owners, name, namespace, param)
        if alias in bindings:
            raise ProtocolError("Tool names must be unique within each namespace", param=param)
        description = tool.get("description")
        if description is None:
            description = ""
        description = require_string(description, param + ".description", empty=True)
        if namespace is not None:
            description = f"Namespace: {namespace}. Tool: {name}.\n{namespace_description}\n{description}".strip()
        if kind == "function":
            definition = tool_definition(
                alias,
                description,
                tool.get("parameters") if tool.get("parameters") is not None else {"type": "object", "properties": {}},
                param,
            )
        else:
            format_spec = tool.get("format") or {"type": "text"}
            require_object(format_spec, param + ".format")
            if require_string(format_spec.get("type"), param + ".format.type") not in {"text", "grammar"}:
                raise ProtocolError("Unsupported custom tool format", param=param + ".format")
            if format_spec.get("type") == "grammar":
                grammar = require_string(format_spec.get("definition"), param + ".format.definition")
                description += "\nThe input string must follow this grammar (best effort):\n" + grammar
            description += "\nReturn the complete raw tool input as the JSON string property 'input'."
            definition = tool_definition(
                alias,
                description,
                {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"], "additionalProperties": False},
                param,
            )
        tools.append(definition)
        bindings[alias] = ResponseTool(name, kind, namespace)

    for index, tool in enumerate(req.tools):
        param = f"tools.{index}"
        if tool.get("type") != "namespace":
            add_tool(tool, param)
            continue
        namespace = require_string(tool.get("name"), param + ".name")
        if namespace in namespaces:
            raise ProtocolError("Namespace names must be unique", param=param + ".name")
        namespaces.add(namespace)
        description = require_string(tool.get("description", ""), param + ".description", empty=True)
        if "tools" in tool and "children" in tool:
            raise ProtocolError("Supply namespace tools only once, not both tools and children", param=param)
        # tools is the Responses format; children is an older client spelling.
        field = "tools" if "tools" in tool else "children"
        children = tool.get(field)
        if not isinstance(children, list):
            raise ProtocolError("Namespace tools must be an array of function/custom tools", param=param + "." + field)
        for child_index, raw in enumerate(children):
            child_param = f"{param}.{field}.{child_index}"
            add_tool(require_object(raw, child_param), child_param, namespace, description)
    validate_tool_names(tools)
    return tools, bindings


def _tool_choice(value: Any, tools: list[dict], bindings: dict[str, ResponseTool]) -> tuple[list[dict], Any]:
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return tools, value
    choice = require_object(value, "tool_choice")

    def select(selector: Any, param: str) -> set[str]:
        selector = require_object(selector, param)
        kind = require_string(selector.get("type"), param + ".type")
        name = require_string(selector.get("name"), param + ".name")
        if kind == "namespace":
            selected = {alias for alias, tool in bindings.items() if tool.namespace == name}
        elif kind in {"function", "custom"}:
            namespace = _tool_namespace(selector.get("namespace"), param + ".namespace")
            alias = _tool_backend_name(name, namespace)
            selected = {alias} if bindings.get(alias) == ResponseTool(name, kind, namespace) else set()
        else:
            raise ProtocolError("Only client function/custom tools or a namespace may be selected", param=param)
        if not selected:
            raise ProtocolError("tool_choice names a tool or namespace that was not supplied", param=param)
        return selected

    kind = require_string(choice.get("type"), "tool_choice.type")
    if kind in {"function", "custom"}:
        alias = next(iter(select(choice, "tool_choice")))
        return tools, {"type": "function", "function": {"name": alias}}
    if kind == "namespace":
        selected, mode = select(choice, "tool_choice"), "required"
    elif kind == "allowed_tools":
        mode = choice.get("mode")
        if mode not in ("auto", "required"):
            raise ProtocolError("allowed_tools mode must be auto or required", param="tool_choice.mode")
        allowed = choice.get("tools")
        if not isinstance(allowed, list) or not allowed:
            raise ProtocolError("allowed_tools must contain at least one tool selector", param="tool_choice.tools")
        selected = set()
        for index, selector in enumerate(allowed):
            selected.update(select(selector, f"tool_choice.tools.{index}"))
    else:
        raise ProtocolError("Unsupported tool_choice type", param="tool_choice")
    # Enforce a namespace/allowlist through the actual backend tool set, not a
    # hint that could let it invoke a tool outside the client's restriction.
    return [tool for tool in tools if tool["function"]["name"] in selected], mode


def prepare_request(req: ResponsesRequest, store: ResponseStore) -> tuple[dict, list[dict], dict[str, ResponseTool]]:
    for field in ("background", "conversation", "prompt", "context_management", "max_tool_calls", "top_logprobs"):
        if getattr(req, field):
            raise ProtocolError(f"{field} is not supported by this local adapter", param=field, code="unsupported_parameter")
    if req.truncation != "disabled":
        raise ProtocolError("Automatic Responses truncation is not supported", param="truncation")
    if req.service_tier not in {None, "auto", "default"}:
        raise ProtocolError("Paid service tiers are not available", param="service_tier")
    if req.prompt_cache_retention not in {None, "in_memory"}:
        raise ProtocolError("Persistent prompt caching is not available", param="prompt_cache_retention")
    if any(field != "reasoning.encrypted_content" for field in req.include):
        raise ProtocolError("Only reasoning.encrypted_content is accepted in include (no encrypted content is produced)", param="include")
    tools, bindings = _tools(req)
    history = store.get(req.previous_response_id)[1] if req.previous_response_id else []
    history += input_messages(req.input)
    messages = ([{"role": "system", "content": req.instructions}] if req.instructions else []) + _backend_history(history, bindings)
    if not messages:
        raise ProtocolError("Provide input or previous_response_id", param="input")
    tools, choice = _tool_choice(req.tool_choice, tools, bindings)
    validate_tool_choice(choice, tools)
    thinking = req.thinking
    if req.reasoning:
        effort = req.reasoning.get("effort")
        if effort is not None and (
            not isinstance(effort, str) or effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
        ):
            raise ProtocolError("Unsupported reasoning effort", param="reasoning.effort")
        if thinking is None and effort is not None:
            thinking = effort != "none"
    response_format = None
    if req.text:
        fmt = require_object(req.text.get("format", {"type": "text"}), "text.format")
        if require_string(fmt.get("type"), "text.format.type") in {"text", "json_object"}:
            response_format = fmt
        elif fmt.get("type") == "json_schema":
            require_object(fmt.get("schema"), "text.format.schema")
            response_format = {"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}}
        else:
            raise ProtocolError("Unsupported text format", param="text.format")
    payload = {
        "model": req.model,
        "messages": messages,
        "stream": req.stream,
        "tools": tools,
        "tool_choice": choice,
        "parallel_tool_calls": req.parallel_tool_calls,
        "thinking": thinking,
        "search": req.search,
        "session_id": req.session_id,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "user": req.user,
        "response_format": response_format,
        "stream_options": {"include_usage": True},
    }
    return payload, history, bindings


class ResponseEvents:
    def __init__(self, req: ResponsesRequest, bindings: dict[str, ResponseTool]):
        self.bindings, self.sequence, self.current = bindings, 0, None
        self.response = {
            "id": uid("resp_"),
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "background": False,
            "error": None,
            "incomplete_details": None,
            "instructions": req.instructions,
            "max_output_tokens": req.max_output_tokens,
            "model": req.model,
            "output": [],
            "parallel_tool_calls": req.parallel_tool_calls,
            "previous_response_id": req.previous_response_id,
            "reasoning": req.reasoning,
            "store": req.store,
            "temperature": req.temperature if req.temperature is not None else 1.0,
            "text": req.text or {"format": {"type": "text"}},
            "tool_choice": req.tool_choice,
            "tools": req.tools,
            "top_p": req.top_p if req.top_p is not None else 1.0,
            "truncation": "disabled",
            "usage": None,
            "metadata": req.metadata,
        }

    def event(self, kind: str, **data) -> str:
        event = sse(kind, {"type": kind, "sequence_number": self.sequence, "response_id": self.response["id"], **data})
        self.sequence += 1
        return event

    def start(self) -> list[str]:
        return [self.event("response.created", response=self.response), self.event("response.in_progress", response=self.response)]

    def close_part(self, status: str = "completed") -> list[str]:
        if self.current is None:
            return []
        kind, index = self.current
        item = self.response["output"][index]
        data = {"item_id": item["id"], "output_index": index}
        if kind == "text":
            part = item["content"][0]
            events = [
                self.event("response.output_text.done", **data, content_index=0, text=part["text"], logprobs=[]),
                self.event("response.content_part.done", **data, content_index=0, part=part),
            ]
        else:
            part = item["summary"][0]
            events = [
                self.event("response.reasoning_summary_text.done", **data, summary_index=0, text=part["text"]),
                self.event("response.reasoning_summary_part.done", **data, summary_index=0, part=part),
            ]
        item["status"] = status
        events.append(self.event("response.output_item.done", output_index=index, item=item))
        self.current = None
        return events

    def feed(self, kind: str, value: Any) -> list[str]:
        events = []
        if kind == "tool" or (self.current is not None and self.current[0] != kind):
            events += self.close_part()
        if kind == "tool":
            fn = value["function"]
            tool = self.bindings[fn["name"]]
            is_custom = tool.kind == "custom"
            field_name = "input" if is_custom else "arguments"
            tool_input = fn["arguments"]
            if is_custom:
                tool_input = json.loads(tool_input).get("input")
                if not isinstance(tool_input, str):
                    raise ProtocolError("Custom tool output must contain a string input", 502)
            item = {
                "type": "custom_tool_call" if is_custom else "function_call",
                "id": uid("ctc_" if is_custom else "fc_"),
                "call_id": value["id"],
                "name": tool.name,
                field_name: "",
                "status": "in_progress",
            }
            if tool.namespace is not None:
                item["namespace"] = tool.namespace
            index = len(self.response["output"])
            self.response["output"].append(item)
            events.append(self.event("response.output_item.added", output_index=index, item=item))
            prefix = "response.custom_tool_call_input" if is_custom else "response.function_call_arguments"
            data = {"item_id": item["id"], "output_index": index}
            if tool_input:
                events.append(self.event(prefix + ".delta", **data, delta=tool_input))
            item[field_name], item["status"] = tool_input, "completed"
            events.append(self.event(prefix + ".done", **data, name=tool.name, **{field_name: tool_input}))
            events.append(self.event("response.output_item.done", output_index=index, item=item))
            return events
        if self.current is None:
            index = len(self.response["output"])
            item = (
                {"type": "message", "id": uid("msg_"), "role": "assistant", "content": [], "status": "in_progress"}
                if kind == "text"
                else {"type": "reasoning", "id": uid("rs_"), "summary": [], "status": "in_progress"}
            )
            self.response["output"].append(item)
            events.append(self.event("response.output_item.added", output_index=index, item=item))
            data = {"item_id": item["id"], "output_index": index}
            if kind == "text":
                part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                item["content"].append(part)
                events.append(self.event("response.content_part.added", **data, content_index=0, part=part))
            else:
                part = {"type": "summary_text", "text": ""}
                item["summary"].append(part)
                events.append(self.event("response.reasoning_summary_part.added", **data, summary_index=0, part=part))
            self.current = (kind, index)
        _, index = self.current
        item = self.response["output"][index]
        data = {"item_id": item["id"], "output_index": index}
        if kind == "text":
            item["content"][0]["text"] += value
            events.append(self.event("response.output_text.delta", **data, content_index=0, delta=value, logprobs=[]))
        else:
            item["summary"][0]["text"] += value
            events.append(self.event("response.reasoning_summary_text.delta", **data, summary_index=0, delta=value))
        return events

    def finish(self, run: ChatRun, req: ResponsesRequest, history: list[dict], store: ResponseStore) -> list[str]:
        incomplete = run.finish_reason in {"length", "content_filter"}
        events = self.close_part("incomplete" if incomplete else "completed")
        self.response["status"] = "incomplete" if incomplete else "completed"
        if incomplete:
            self.response["incomplete_details"] = {"reason": "max_output_tokens" if run.finish_reason == "length" else "content_filter"}
        self.response["usage"] = run.token_usage()
        if run.session_id:
            self.response["session_id"] = run.session_id
        if req.store:
            self.response["store"] = store.put(self.response, history + [_canonical_assistant(run.assistant_message(), self.bindings)])
        events.append(self.event("response." + self.response["status"], response=self.response))
        return events


def create_responses_router(backend: Backend, store: ResponseStore | None = None) -> APIRouter:
    router = APIRouter(route_class=NativeRoute, tags=["Responses"])
    cache = store if store is not None else ResponseStore()

    @router.post("/v1/responses")
    async def create_response(req: ResponsesRequest):
        payload, history, bindings = prepare_request(req, cache)
        upstream = await backend(payload)
        run = ChatRun(upstream, payload, req.max_output_tokens)
        events = ResponseEvents(req, bindings)
        if not req.stream:
            async for kind, value in run:
                events.feed(kind, value)
            events.finish(run, req, history, cache)
            return events.response

        async def generate():
            iterator = run.__aiter__()
            try:
                for event in events.start():
                    yield event
                async for kind, value in iterator:
                    for event in events.feed(kind, value):
                        yield event
                for event in events.finish(run, req, history, cache):
                    yield event
            except Exception as exc:  # noqa: BLE001 - protocol boundary; unexpected errors are logged
                error = error_from_exception(exc)
                for event in events.close_part("incomplete"):
                    yield event
                events.response["status"] = "failed"
                events.response["error"] = {"code": error.code or "server_error", "message": error.message}
                events.response["store"] = False
                events.response["usage"] = run.token_usage()
                yield events.event("error", code=error.code or "server_error", message=error.message, param=error.param)
                yield events.event("response.failed", response=events.response)
            finally:
                await iterator.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)

    @router.get("/v1/responses/{response_id}")
    async def retrieve_response(response_id: str):
        return cache.get(response_id)[0]

    @router.delete("/v1/responses/{response_id}")
    async def delete_response(response_id: str):
        cache.get(response_id)
        cache.delete(response_id)
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    return router
