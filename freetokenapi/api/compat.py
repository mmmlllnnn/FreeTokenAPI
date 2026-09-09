"""Shared, in-process adapters for the native Responses and Messages protocols.

No network requests or client-side tools are executed here. The existing chat
backend owns upstream connections, account selection, and tool emulation.
"""

from __future__ import annotations

import codecs
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from ..tokens import count_messages_tokens, estimate_tokens

Backend = Callable[[dict[str, Any]], Awaitable[Any]]
log = logging.getLogger("freetokenapi.api.compat")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def uid(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class ProtocolError(Exception):
    def __init__(self, message: str, status: int = 400, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message, self.status, self.param, self.code = message, status, param, code


def error_from_exception(exc: Exception) -> ProtocolError:
    if isinstance(exc, ProtocolError):
        return exc
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except ValueError:
                pass
        if isinstance(detail, dict):
            detail = detail.get("error", detail)
            if isinstance(detail, dict):
                detail = detail.get("message", "Upstream request failed")
        return ProtocolError(str(detail), exc.status_code)
    log.exception("native protocol request failed", exc_info=exc)
    return ProtocolError("Internal adapter error; see the local server log", 500)


def error_body(error: ProtocolError, protocol: str) -> dict:
    types = {
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        529: "overloaded_error",
    }
    kind = types.get(error.status, "invalid_request_error" if error.status < 500 else "api_error")
    if protocol == "messages":
        return {"type": "error", "error": {"type": kind, "message": error.message}}
    return {
        "error": {
            "message": error.message,
            "type": "server_error" if error.status >= 500 else ("rate_limit_error" if error.status == 429 else "invalid_request_error"),
            "param": error.param,
            "code": error.code,
        }
    }


class NativeRoute(APIRoute):
    protocol = "responses"

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            request_id = uid("req_")
            try:
                response = await original(request)
            except RequestValidationError as exc:
                # Never include Pydantic's input field: it may contain private prompts.
                issue = exc.errors()[0]
                param = ".".join(str(p) for p in issue["loc"] if p != "body")
                error = ProtocolError(f"Invalid {param or 'request'}: {issue['msg']}", param=param or None)
                response = JSONResponse(error_body(error, self.protocol), status_code=400)
            except Exception as exc:  # noqa: BLE001 - protocol boundary; unexpected errors are logged
                error = error_from_exception(exc)
                response = JSONResponse(
                    error_body(error, self.protocol),
                    status_code=error.status,
                    headers=exc.headers if isinstance(exc, HTTPException) else None,
                )
            response.headers["request-id" if self.protocol == "messages" else "x-request-id"] = request_id
            return response

        return handler


class ChatRoute(NativeRoute):
    """Chat Completions uses the same OpenAI error envelope as Responses."""
    protocol = "chat.completions"


class MessagesRoute(NativeRoute):
    protocol = "messages"


class NativeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Client/proxy tracing hints are accepted only for wire compatibility.
    # Never forward them as prompts/settings or include them in stored payloads.
    client_metadata: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)


def require_string(value: Any, param: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ProtocolError(f"{param} must be {'a string' if empty else 'a non-empty string'}", param=param)
    return value


def require_object(value: Any, param: str) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError(f"{param} must be an object", param=param)
    return value


def image_part(uri: Any, param: str) -> dict:
    uri = require_string(uri, param)
    if not uri.startswith("data:image/") or ";base64," not in uri:
        raise ProtocolError("Only base64 data:image/... inputs are supported; remote images and file IDs are not fetched", param=param)
    return {"type": "image_url", "image_url": {"url": uri}}


def file_part(data: Any, filename: Any, param: str) -> dict:
    data = require_string(data, param + ".file_data")
    filename = require_string(filename, param + ".filename")
    if data.startswith(("http://", "https://", "file:")):
        raise ProtocolError("Attachments must contain inline base64 data; remote URLs and file IDs are not fetched", param=param)
    return {"type": "file", "file": {"filename": filename, "file_data": data}}


def text_content(parts: list[dict]) -> str | list[dict]:
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def tool_definition(name: Any, description: Any, schema: Any, param: str) -> dict:
    name = require_string(name, param + ".name")
    schema = require_object(schema, param + ".parameters")
    if schema.get("type", "object") != "object":
        raise ProtocolError("Tool parameters must use an object schema", param=param)
    if description is not None and not isinstance(description, str):
        raise ProtocolError("Tool description must be a string", param=param)
    return {"type": "function", "function": {"name": name, "description": description or "", "parameters": schema}}


def validate_tool_names(tools: list[dict]) -> None:
    names = [tool["function"]["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise ProtocolError("Tool names must be unique", param="tools")


def validate_tool_choice(choice: Any, tools: list[dict]) -> None:
    if choice == "required" and not tools:
        raise ProtocolError("tool_choice requires at least one tool", param="tool_choice")
    if isinstance(choice, dict):
        name = choice["function"]["name"]
        if name not in {tool["function"]["name"] for tool in tools}:
            raise ProtocolError("tool_choice names a tool that was not supplied", param="tool_choice")


def prompt_tokens(payload: dict) -> int:
    return count_messages_tokens(payload.get("messages", [])) + estimate_tokens(dumps(payload["tools"]) if payload.get("tools") else "")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {dumps(data)}\n\n"


async def close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


async def chat_chunks(response: Any) -> AsyncIterator[dict]:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if response.get("error"):
            yield response
            return
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise ProtocolError("Backend returned an invalid chat completion", 502)
        choice = choices[0]
        yield {
            "choices": [{"delta": choice["message"], "finish_reason": choice.get("finish_reason")}],
            "usage": response.get("usage"),
            "session_id": response.get("session_id"),
        }
        return
    if not isinstance(response, StreamingResponse):
        raise ProtocolError("Backend returned an unsupported response", 502)
    source = response.body_iterator
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    def decode_frame(frame: str):
        lines = [line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:")]
        if not lines:
            return None
        data = "\n".join(lines)
        if data == "[DONE]":
            return data
        try:
            result = json.loads(data)
        except ValueError as exc:
            raise ProtocolError("Backend sent malformed SSE JSON", 502) from exc
        if not isinstance(result, dict):
            raise ProtocolError("Backend SSE data must be an object", 502)
        return result

    try:
        async for chunk in source:
            buffer += decoder.decode(bytes(chunk)) if isinstance(chunk, (bytes, bytearray, memoryview)) else chunk
            while True:
                boundary = re.search(r"\r\n\r\n|\n\n|\r\r", buffer)
                if boundary is None:
                    break
                frame, buffer = buffer[: boundary.start()], buffer[boundary.end() :]
                data = decode_frame(frame)
                if data == "[DONE]":
                    return
                if data is not None:
                    yield data
            if len(buffer) > 8 * 1024 * 1024:
                raise ProtocolError("Backend SSE event exceeds the local size limit", 502)
        buffer += decoder.decode(b"", final=True)
        if buffer.strip():
            data = decode_frame(buffer)
            if data == "[DONE]":
                return
            if data is not None:
                yield data
        raise ProtocolError("Backend stream ended before [DONE]", 502)
    finally:
        await close_iterator(source)


class OutputLimit:
    """Local, approximate token limit and boundary-safe stop sequence matching.

    The web backends do not expose native token budgets. One CJK character costs
    one estimated token; four other characters cost one. Tool JSON is atomic.
    """

    def __init__(self, max_tokens: int | None, stops: list[str] | None = None):
        self.remaining = max_tokens * 4 if max_tokens is not None else None
        self.stops = stops or []
        self.pending = ""
        self.reason: str | None = None
        self.stop_sequence: str | None = None

    def take(self, text: str, *, atomic: bool = False) -> str:
        if self.reason:
            return ""
        units = len(text) + 3 * len(_CJK.findall(text))
        if self.remaining is None:
            return text
        if units <= self.remaining:
            self.remaining -= units
            return text
        self.reason = "length"
        if atomic:
            return ""
        end = 0
        for char in text:
            cost = 4 if _CJK.fullmatch(char) else 1
            if cost > self.remaining:
                break
            self.remaining -= cost
            end += 1
        return text[:end]

    def text(self, delta: str, *, final: bool = False) -> str:
        if self.reason:
            return ""
        text, self.pending = self.pending + delta, ""
        matches = [(text.find(stop), index, stop) for index, stop in enumerate(self.stops) if stop in text]
        if matches:
            position, _, stop = min(matches)
            result = self.take(text[:position])
            if not self.reason:
                self.reason, self.stop_sequence = "stop", stop
            return result
        if not final:
            keep = max((n for stop in self.stops for n in range(1, min(len(stop), len(text) + 1)) if text.endswith(stop[:n])), default=0)
            if keep:
                text, self.pending = text[:-keep], text[-keep:]
        return self.take(text)


@dataclass
class ChatRun:
    response: Any
    payload: dict
    max_tokens: int | None = None
    stops: list[str] | None = None
    parts: list[tuple[str, Any]] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    stop_sequence: str | None = None
    session_id: str | None = None

    def remember(self, kind: str, value: Any) -> tuple[str, Any]:
        if kind != "tool" and self.parts and self.parts[-1][0] == kind:
            self.parts[-1] = (kind, self.parts[-1][1] + value)
        else:
            self.parts.append((kind, value))
        return kind, value

    def token_usage(self) -> dict:
        output = "".join(value["function"]["arguments"] if kind == "tool" else value for kind, value in self.parts)
        reasoning = "".join(value for kind, value in self.parts if kind == "reasoning")
        input_count = self.usage.get("prompt_tokens", prompt_tokens(self.payload))
        output_count = self.usage.get("completion_tokens", estimate_tokens(output))
        cached = (self.usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        reasoning_count = (self.usage.get("completion_tokens_details") or {}).get("reasoning_tokens", estimate_tokens(reasoning))
        return {
            "input_tokens": input_count,
            "output_tokens": output_count,
            "total_tokens": input_count + output_count,
            "input_tokens_details": {"cached_tokens": min(cached, input_count), "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": min(reasoning_count, output_count)},
        }

    def assistant_message(self) -> dict:
        message: dict = {"role": "assistant", "content": "".join(v for k, v in self.parts if k == "text")}
        calls = [v for k, v in self.parts if k == "tool"]
        if calls:
            message["tool_calls"] = calls
        return message

    async def __aiter__(self):
        limiter = OutputLimit(self.max_tokens, self.stops)
        calls: dict[int, dict] = {}
        finished = False
        chunks = chat_chunks(self.response)
        try:
            async for chunk in chunks:
                if chunk.get("error"):
                    error = chunk["error"]
                    message = error.get("message", "Backend stream failed") if isinstance(error, dict) else str(error)
                    raise ProtocolError(message, 502)
                if isinstance(chunk.get("usage"), dict):
                    self.usage = chunk["usage"]
                if chunk.get("session_id"):
                    self.session_id = chunk["session_id"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                for kind, key in (("reasoning", "reasoning_content"), ("text", "content")):
                    text = delta.get(key)
                    if text is None or text == "":
                        continue
                    if not isinstance(text, str):
                        raise ProtocolError("Backend returned non-text output", 502)
                    value = limiter.text(text) if kind == "text" else limiter.take(text)
                    if value:
                        yield self.remember(kind, value)
                if limiter.reason:
                    break
                for position, fragment in enumerate(delta.get("tool_calls") or []):
                    index = fragment.get("index", position)
                    if not isinstance(index, int) or index < 0:
                        raise ProtocolError("Backend returned an invalid tool index", 502)
                    call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if fragment.get("id"):
                        call["id"] = fragment["id"]
                    function = fragment.get("function") or {}
                    for key in ("name", "arguments"):
                        value = function.get(key)
                        if value is not None:
                            if not isinstance(value, str):
                                raise ProtocolError("Backend returned invalid tool call fields", 502)
                            call["function"][key] += value
                if choice.get("finish_reason") is not None:
                    self.finish_reason = choice["finish_reason"]
                    finished = True
            if not limiter.reason and not finished:
                raise ProtocolError("Backend returned no completion finish reason", 502)
            tail = limiter.text("", final=True)
            if tail:
                yield self.remember("text", tail)
            if not limiter.reason:
                if self.finish_reason not in {"stop", "length", "content_filter", "tool_calls", "function_call"}:
                    raise ProtocolError("Backend reported an incomplete or failed completion", 502)
                if self.finish_reason in {"tool_calls", "function_call"} and not calls:
                    raise ProtocolError("Backend reported a tool call without tool data", 502)
                if self.payload.get("parallel_tool_calls") is False and len(calls) > 1:
                    raise ProtocolError("Backend returned parallel tools despite the one-tool limit", 502)
                choice = self.payload.get("tool_choice", "auto")
                if choice == "none" and calls:
                    raise ProtocolError("Backend returned a tool when tool_choice was none", 502)
                if (choice == "required" or isinstance(choice, dict)) and not calls and self.finish_reason == "stop":
                    raise ProtocolError("Backend did not produce the required tool call", 502)
                if isinstance(choice, dict) and any(call["function"]["name"] != choice["function"]["name"] for call in calls.values()):
                    raise ProtocolError("Backend returned a different tool than requested", 502)
                names = {tool["function"]["name"] for tool in self.payload.get("tools") or []}
                for index in sorted(calls):
                    call = calls[index]
                    function = call["function"]
                    if not function["name"] or function["name"] not in names:
                        raise ProtocolError("Backend called an undeclared tool", 502)
                    try:
                        arguments = json.loads(function["arguments"] or "{}")
                    except ValueError as exc:
                        raise ProtocolError("Backend returned invalid tool arguments JSON", 502) from exc
                    if not isinstance(arguments, dict):
                        raise ProtocolError("Backend tool arguments must be a JSON object", 502)
                    function["arguments"] = function["arguments"] or "{}"
                    call["id"] = call["id"] or uid("call_")
                # Validate and budget the whole batch before exposing executable
                # tool calls to a client. Never emit half of malformed tool JSON.
                batch_text = "".join(call["function"]["name"] + call["function"]["arguments"] for call in calls.values())
                if calls and limiter.take(batch_text, atomic=True):
                    if self.finish_reason == "stop":
                        self.finish_reason = "tool_calls"
                    for index in sorted(calls):
                        yield self.remember("tool", calls[index])
            if limiter.reason:
                self.finish_reason = limiter.reason
                self.stop_sequence = limiter.stop_sequence
        finally:
            # Explicitly close the nested iterator, including on disconnect/limit.
            # This releases the original account semaphore and upstream response.
            await close_iterator(chunks)
