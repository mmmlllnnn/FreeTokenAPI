"""Anthropic Messages wire format backed by the existing web-model chat API."""

from __future__ import annotations

import base64
import json
import mimetypes
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

from .compat import (
    SSE_HEADERS,
    Backend,
    ChatRun,
    MessagesRoute,
    NativeRequest,
    ProtocolError,
    dumps,
    error_body,
    error_from_exception,
    file_part,
    image_part,
    prompt_tokens,
    require_object,
    require_string,
    sse,
    text_content,
    tool_definition,
    uid,
    validate_tool_choice,
    validate_tool_names,
)


class CountTokensRequest(NativeRequest):
    model: str = Field(min_length=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None
    # Common client clearing hints are accepted, but cloud edits are not applied.
    context_management: dict[str, Any] | None = None
    # Unsupported server facilities are named so errors explain the limitation.
    mcp_servers: list[dict[str, Any]] | None = None
    container: Any = None


class MessagesRequest(CountTokensRequest):
    max_tokens: int = Field(gt=0)
    stream: bool = False
    stop_sequences: list[str] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0, le=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    output_format: dict[str, Any] | None = None
    # FreeTokenAPI extensions; ordinary Claude clients need not supply these.
    search: bool | None = None
    session_id: str | None = None


def _parts(value: Any, param: str) -> list[dict]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        raise ProtocolError("Content must be a string or an array of blocks", param=param)
    parts = []
    for raw in value:
        part = require_object(raw, param)
        kind = require_string(part.get("type"), param + ".type")
        if kind == "text":
            parts.append({"type": "text", "text": require_string(part.get("text"), param + ".text", empty=True)})
        elif kind == "image":
            source = require_object(part.get("source"), param + ".source")
            if source.get("type") != "base64":
                raise ProtocolError("Only base64 image sources are supported; URLs are not fetched", param=param)
            mime = require_string(source.get("media_type"), param + ".source.media_type")
            data = require_string(source.get("data"), param + ".source.data")
            parts.append(image_part(f"data:{mime};base64,{data}", param))
        elif kind == "document":
            source = require_object(part.get("source"), param + ".source")
            source_type = require_string(source.get("type"), param + ".source.type")
            if source_type not in {"base64", "text"}:
                raise ProtocolError("Documents require inline base64 or text sources; URLs and file IDs are not fetched", param=param)
            mime = require_string(source.get("media_type", "text/plain" if source_type == "text" else "application/pdf"), param + ".source.media_type")
            data = require_string(source.get("data"), param + ".source.data")
            if source_type == "text":
                if mime != "text/plain":
                    raise ProtocolError("Text document sources require text/plain", param=param)
                data = base64.b64encode(data.encode("utf-8")).decode("ascii")
            extension = mimetypes.guess_extension(mime) or ".bin"
            filename = part.get("filename") or part.get("title") or ("attachment" + extension)
            filename = require_string(filename, param + ".filename")
            if "." not in filename:
                filename += extension
            parts.append(file_part(f"data:{mime};base64,{data}", filename, param))
        else:
            raise ProtocolError(f"Unsupported Messages content block: {kind}", param=param)
    return parts


def _flush_message(result: list[dict], role: str, pending: list[dict], calls: list[dict]) -> None:
    if pending or calls:
        # Copy before clearing: multimodal text_content returns the input list.
        item: dict = {"role": role, "content": text_content(list(pending))}
        if calls:
            item["tool_calls"] = list(calls)
        result.append(item)
        pending.clear()
        calls.clear()


def message_history(req: CountTokensRequest) -> list[dict]:
    result = []
    if req.system is not None:
        system = _parts(req.system, "system")
        if any(part["type"] != "text" for part in system):
            raise ProtocolError("system accepts text blocks only", param="system")
        result.append({"role": "system", "content": text_content(system)})
    conversation_started = False
    for index, message in enumerate(req.messages):
        param = f"messages.{index}"
        role = require_string(message.get("role"), param + ".role")
        if role in {"system", "developer"}:
            # Client/proxy compatibility: preserve interleaved instructions in
            # place. Moving them to the prefix changes the meaning of history.
            parts = _parts(message.get("content"), param + ".content")
            if any(part["type"] != "text" for part in parts):
                raise ProtocolError("System/developer instructions accept text blocks only", param=param + ".content")
            instructions = text_content(parts)
            if conversation_started:
                result.append({"role": "system", "content": instructions})
            elif result:
                result[0]["content"] += "\n" + instructions
            else:
                result.append({"role": "system", "content": instructions})
            continue
        if role not in {"user", "assistant"}:
            raise ProtocolError(
                f"Unsupported {param}.role: {role!r}. Use user/assistant messages and tool_result blocks for tool outputs.",
                param=param + ".role",
            )
        conversation_started = True
        content = message.get("content")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ProtocolError("Message content must be a string or an array", param=param + ".content")
        pending: list[dict] = []
        calls: list[dict] = []

        for raw in content:
            block = require_object(raw, param + ".content")
            kind = require_string(block.get("type"), param + ".type")
            if kind == "tool_use":
                if role != "assistant":
                    raise ProtocolError("tool_use blocks belong to assistant messages", param=param)
                calls.append(
                    {
                        "id": require_string(block.get("id"), param + ".id"),
                        "type": "function",
                        "function": {
                            "name": require_string(block.get("name"), param + ".name"),
                            "arguments": dumps(require_object(block.get("input"), param + ".input")),
                        },
                    }
                )
            elif kind == "tool_result":
                if role != "user":
                    raise ProtocolError("tool_result blocks belong to user messages", param=param)
                _flush_message(result, role, pending, calls)
                parts = _parts(block.get("content", ""), param + ".tool_result.content")
                if block.get("is_error"):
                    parts.insert(0, {"type": "text", "text": "[Tool execution error]"})
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": require_string(block.get("tool_use_id"), param + ".tool_use_id"),
                        "content": text_content(parts),
                    }
                )
            elif kind in {"thinking", "redacted_thinking"}:
                if role != "assistant":
                    raise ProtocolError("Thinking blocks belong to assistant messages", param=param)
                # Local web-model reasoning is not an Anthropic signed transcript.
                # Replayed thinking is annotation, not a new prompt or tool call.
            else:
                pending.extend(_parts([block], param + ".content"))
        _flush_message(result, role, pending, calls)
    if not any(message["role"] != "system" for message in result):
        raise ProtocolError("At least one non-empty message is required", param="messages")
    return result


def _context_count(value: Any, param: str, units: set[str], *, minimum: int = 0) -> None:
    value = require_object(value, param)
    unit = require_string(value.get("type"), param + ".type")
    number = value.get("value")
    if unit not in units or type(number) is not int or number < minimum:
        raise ProtocolError("Invalid context-management count or unit", param=param)
    if value.keys() - {"type", "value"}:
        raise ProtocolError("Unsupported context-management count option", param=param)


def _validate_context_management(config: dict | None) -> None:
    """Accept client clearing hints without pretending to perform cloud edits.

    Claude Code sends these hints even for short, ordinary messages. Rejecting
    them prevents otherwise supported requests from ever reaching the backend.
    Preserve normal history conversion and report applied_edits=[] instead of
    silently clearing tool results or claiming Anthropic context compression.
    """
    if config is None:
        return
    if config.keys() - {"edits"}:
        raise ProtocolError("Unsupported context_management option", param="context_management")
    edits = config.get("edits", [])
    if not isinstance(edits, list):
        raise ProtocolError("context_management.edits must be an array", param="context_management.edits")
    for index, raw in enumerate(edits):
        param = f"context_management.edits.{index}"
        edit = require_object(raw, param)
        kind = require_string(edit.get("type"), param + ".type")
        if kind == "clear_thinking_20251015":
            allowed = {"type", "keep"}
            keep = edit.get("keep")
            if keep is not None and keep != "all":
                _context_count(keep, param + ".keep", {"thinking_turns"}, minimum=1)
        elif kind == "clear_tool_uses_20250919":
            allowed = {"type", "trigger", "keep", "clear_at_least", "exclude_tools", "clear_tool_inputs"}
            for field, units in (("trigger", {"input_tokens", "tool_uses"}), ("keep", {"tool_uses"}), ("clear_at_least", {"input_tokens"})):
                if edit.get(field) is not None:
                    _context_count(edit[field], param + "." + field, units)
            if edit.get("clear_tool_inputs") is not None and not isinstance(edit["clear_tool_inputs"], bool):
                raise ProtocolError("clear_tool_inputs must be a boolean", param=param + ".clear_tool_inputs")
            if edit.get("exclude_tools") is not None:
                if not isinstance(edit["exclude_tools"], list):
                    raise ProtocolError("exclude_tools must be an array", param=param + ".exclude_tools")
                for name in edit["exclude_tools"]:
                    require_string(name, param + ".exclude_tools")
        else:
            raise ProtocolError(
                f"Unsupported context edit: {kind}. Only thinking/tool clearing hints are accepted; "
                "server-side compaction is unavailable. Send client-managed message history instead.",
                param=param + ".type",
            )
        if edit.keys() - allowed:
            raise ProtocolError("Unsupported context edit option", param=param)


def _context_headers(req: CountTokensRequest) -> dict[str, str]:
    return {"X-FreeTokenAPI-Context-Management": "not-applied"} if req.context_management is not None else {}


def prepare_request(req: CountTokensRequest) -> dict:
    _validate_context_management(req.context_management)
    for field in ("mcp_servers", "container"):
        if getattr(req, field):
            raise ProtocolError(f"{field} is not supported by this local adapter", param=field)
    tools = []
    for index, tool in enumerate(req.tools):
        param = f"tools.{index}"
        if tool.get("type") is not None and tool.get("type") != "custom":
            raise ProtocolError(
                "Only client tools with an input_schema are supported; Anthropic server tools are not available", param=param
            )
        tools.append(tool_definition(tool.get("name"), tool.get("description"), tool.get("input_schema"), param))
    validate_tool_names(tools)
    choice: Any = "auto"
    parallel = True
    if req.tool_choice:
        kind = require_string(req.tool_choice.get("type"), "tool_choice.type")
        if kind in {"auto", "none"}:
            choice = kind
        elif kind == "any":
            choice = "required"
        elif kind == "tool":
            choice = {"type": "function", "function": {"name": require_string(req.tool_choice.get("name"), "tool_choice.name")}}
        else:
            raise ProtocolError("Unsupported tool_choice type", param="tool_choice")
        parallel = not req.tool_choice.get("disable_parallel_tool_use", False)
    validate_tool_choice(choice, tools)
    thinking = None
    if req.thinking:
        mode = require_string(req.thinking.get("type"), "thinking.type")
        if mode not in {"enabled", "disabled", "adaptive"}:
            raise ProtocolError("Unsupported thinking type", param="thinking.type")
        thinking = mode != "disabled"
    payload = {
        "model": req.model,
        "messages": message_history(req),
        "tools": tools,
        "tool_choice": choice,
        "parallel_tool_calls": parallel,
        "thinking": thinking,
        "stream_options": {"include_usage": True},
    }
    fmt = (req.output_config or {}).get("format")
    if isinstance(req, MessagesRequest):
        if req.service_tier not in {None, "auto", "standard_only"}:
            raise ProtocolError("Paid service tiers are not available", param="service_tier")
        if len(req.stop_sequences) > 16 or any(not stop or len(stop) > 1024 for stop in req.stop_sequences):
            raise ProtocolError("Use at most 16 non-empty stop sequences, each no longer than 1024 characters", param="stop_sequences")
        payload.update(
            stream=req.stream,
            temperature=req.temperature,
            top_p=req.top_p,
            search=req.search,
            session_id=req.session_id,
            user=(req.metadata or {}).get("user_id"),
        )
        fmt = req.output_format or fmt
    if fmt:
        require_object(fmt, "output_config.format")
        if fmt.get("type") != "json_schema":
            raise ProtocolError("Only json_schema output format is supported", param="output_config.format")
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": require_object(fmt.get("schema"), "output_config.format.schema")},
        }
    return payload


def message_usage(run: ChatRun) -> dict:
    usage = run.token_usage()
    cached = usage["input_tokens_details"]["cached_tokens"]
    return {
        "input_tokens": usage["input_tokens"] - cached,
        "output_tokens": usage["output_tokens"],
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


class MessageEvents:
    def __init__(self, req: MessagesRequest, payload: dict):
        self.current: tuple[str, int] | None = None
        self.show_thinking = (req.thinking or {}).get("display") != "omitted"
        self.message = {
            "id": uid("msg_"),
            "type": "message",
            "role": "assistant",
            "model": req.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens(payload),
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

        if req.context_management is not None:
            self.message["context_management"] = {"applied_edits": []}

    @staticmethod
    def event(kind: str, **data) -> str:
        return sse(kind, {"type": kind, **data})

    def close_part(self) -> list[str]:
        if self.current is None:
            return []
        kind, index = self.current
        self.current = None
        events = []
        if kind == "reasoning":
            # Empty by design: do not pretend to possess Anthropic signing keys.
            events.append(self.event("content_block_delta", index=index, delta={"type": "signature_delta", "signature": ""}))
        events.append(self.event("content_block_stop", index=index))
        return events

    def feed(self, kind: str, value: Any) -> list[str]:
        if kind == "reasoning" and not self.show_thinking:
            return []
        events = []
        if kind == "tool" or (self.current is not None and self.current[0] != kind):
            events += self.close_part()
        if kind == "tool":
            index = len(self.message["content"])
            block = {"type": "tool_use", "id": value["id"], "name": value["function"]["name"], "input": {}}
            events.append(self.event("content_block_start", index=index, content_block=block))
            arguments = value["function"]["arguments"]
            events.append(self.event("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": arguments}))
            block["input"] = json.loads(arguments)
            self.message["content"].append(block)
            events.append(self.event("content_block_stop", index=index))
            return events
        if self.current is None:
            index = len(self.message["content"])
            block = {"type": "text", "text": ""} if kind == "text" else {"type": "thinking", "thinking": "", "signature": ""}
            self.message["content"].append(block)
            events.append(self.event("content_block_start", index=index, content_block=block))
            self.current = (kind, index)
        _, index = self.current
        field = "text" if kind == "text" else "thinking"
        self.message["content"][index][field] += value
        if value:
            events.append(self.event("content_block_delta", index=index, delta={"type": field + "_delta", field: value}))
        return events

    def finish(self, run: ChatRun) -> list[str]:
        events = []
        if not self.message["content"]:
            events += self.feed("text", "")
        events += self.close_part()
        stop_reason = {"length": "max_tokens", "content_filter": "refusal", "tool_calls": "tool_use", "function_call": "tool_use"}.get(
            run.finish_reason, "end_turn"
        )
        if run.stop_sequence:
            stop_reason = "stop_sequence"
        self.message.update(stop_reason=stop_reason, stop_sequence=run.stop_sequence, usage=message_usage(run))
        if run.session_id:
            self.message["session_id"] = run.session_id
        events.append(
            self.event(
                "message_delta",
                delta={"stop_reason": stop_reason, "stop_sequence": run.stop_sequence},
                usage=self.message["usage"],
                **({"context_management": self.message["context_management"]} if "context_management" in self.message else {}),
            )
        )
        events.append(self.event("message_stop"))
        return events


def create_messages_router(backend: Backend) -> APIRouter:
    router = APIRouter(route_class=MessagesRoute, tags=["Claude Messages"])

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(req: CountTokensRequest):
        payload = prepare_request(req)
        return JSONResponse(
            {"input_tokens": prompt_tokens(payload)}, headers={"X-FreeTokenAPI-Token-Count": "estimate", **_context_headers(req)}
        )

    @router.post("/v1/messages")
    async def create_message(req: MessagesRequest):
        payload = prepare_request(req)
        upstream = await backend(payload)
        run = ChatRun(upstream, payload, req.max_tokens, req.stop_sequences)
        events = MessageEvents(req, payload)
        if not req.stream:
            async for kind, value in run:
                events.feed(kind, value)
            events.finish(run)
            return JSONResponse(events.message, headers=_context_headers(req))

        async def generate():
            iterator = run.__aiter__()
            try:
                yield events.event("message_start", message=events.message)
                async for kind, value in iterator:
                    for event in events.feed(kind, value):
                        yield event
                for event in events.finish(run):
                    yield event
            except Exception as exc:  # noqa: BLE001 - protocol boundary; unexpected errors are logged
                yield sse("error", error_body(error_from_exception(exc), "messages"))
            finally:
                await iterator.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream", headers={**SSE_HEADERS, **_context_headers(req)})

    return router
