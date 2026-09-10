from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import random
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .. import __version__
from .. import tools as toolemu
from ..accounts import AccountPool, AccountPoolBusy, DeepSeekAccount, account_lock
from ..config import settings
from ..deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekSession
from ..deepseek.models import MODEL_ALIASES, MODEL_TYPE, WebModel, advertised_models
from ..deepseek.stream import IncrementalSSE, MessageReconstructor
from ..qwen import api as qwen_api
from ..qwen.accounts import QwenAccount
from ..qwen.client import QwenClient, QwenError
from ..store import JsonStore
from ..tokens import estimate_tokens
from ..usage import init_tracker, record_usage
from .compat import ChatRoute, close_iterator
from .messages import create_messages_router
from .responses import create_responses_router

log = logging.getLogger("freetokenapi.api")

QWEN_DEFAULT_MODELS = [
    {
        "id": "qwen3.8-max",
        "name": "Qwen3.8-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-plus",
        "name": "Qwen3.7-Plus",
        "owned_by": "qwen",
        "model_type": "chat",
    },
    {
        "id": "qwen3.7-max",
        "name": "Qwen3.7-Max",
        "owned_by": "qwen",
        "model_type": "chat",
    },
]

STATUS_TO_FINISH_REASON = {
    "FINISHED": "stop",
    "CONTEXT_LENGTH_EXCEEDED": "length",
    "CONTENT_FILTER": "content_filter",
    "INCOMPLETE": "response_incomplete",
    "WIP": "response_incomplete",
    "TIMEOUT": "response_incomplete",
}

CONTEXT_LENGTH_STATUS = "CONTEXT_LENGTH_EXCEEDED"
INPUT_EXCEEDS_LIMIT = "input_exceeds_limit"
CONTINUE_PROMPT = "Continue"
MAX_CONTINUE_ROUNDS = 5
RESPONSE_INCOMPLETE = "response_incomplete"
RESPONSE_INCOMPLETE_MESSAGE = "Response is incomplete: provider errors interrupted the continuation, please retry"
REDUCED_CONTEXT_MESSAGE = "Response was generated from reduced context because the original input exceeded the model limit and may be incomplete"


class DeepSeekStreamError(Exception):
    pass


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        # Pi uses developer for reasoning models. Normalize only the role; do
        # not move an instruction or discard any surrounding conversation.
        if v == "developer":
            return "system"
        allowed_roles = {"user", "assistant", "system", "tool", "function"}
        if v not in allowed_roles:
            raise ValueError("Invalid role. Use system, developer, user, assistant, tool, or function")
        return v

    @field_validator("tool_calls", mode="before")
    @classmethod
    def validate_tool_calls(cls, v: list[Any] | None) -> list[dict[str, Any]] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("tool_calls must be a list")
        validated_tool_calls = []
        for tool_call in v:
            if not isinstance(tool_call, dict):
                raise ValueError("Each tool_call must be a dictionary")
            if "function" not in tool_call and "name" not in tool_call:
                raise ValueError("Each tool_call must contain 'function' or 'name'")
            validated_tool_calls.append(tool_call)
        return validated_tool_calls


class FileSpec(BaseModel):
    name: str
    content: str
    content_type: str = "application/octet-stream"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deepseek-web")
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    thinking: bool | None = None
    enable_thinking: bool | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    search: bool | None = None
    session_id: str | None = None
    user: str | None = None
    files: list[FileSpec] | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    response_format: Any = None
    stream_options: Any = None

    @field_validator("thinking", mode="before")
    @classmethod
    def normalize_thinking(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if not isinstance(value.get("type"), str) or value["type"] not in {"enabled", "disabled", "adaptive"}:
            raise ValueError("thinking.type must be enabled, disabled, or adaptive")
        if value.keys() - {"type", "budget_tokens", "clear_thinking"}:
            raise ValueError("Unsupported thinking option")
        if "budget_tokens" in value and (type(value["budget_tokens"]) is not int or value["budget_tokens"] <= 0):
            raise ValueError("thinking.budget_tokens must be a positive integer")
        if "clear_thinking" in value and type(value["clear_thinking"]) is not bool:
            raise ValueError("thinking.clear_thinking must be a boolean")
        # Web backends expose an on/off switch, not official token budgets or
        # provider-specific reasoning-history deletion semantics.
        return value["type"] != "disabled"

    @field_validator("chat_template_kwargs")
    @classmethod
    def validate_chat_template(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value and "enable_thinking" in value and type(value["enable_thinking"]) is not bool:
            raise ValueError("chat_template_kwargs.enable_thinking must be a boolean")
        return value

    @model_validator(mode="after")
    def resolve_thinking(self) -> ChatCompletionRequest:
        if self.thinking is None:
            if self.enable_thinking is not None:
                self.thinking = self.enable_thinking
            elif self.chat_template_kwargs and "enable_thinking" in self.chat_template_kwargs:
                self.thinking = self.chat_template_kwargs["enable_thinking"]
            elif self.reasoning_effort is not None:
                self.thinking = self.reasoning_effort != "none"
        return self


class ImageGenerationRequest(BaseModel):
    model: str = Field(default="qwen-image-gen")
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    response_format: str = Field(default="url")
    session_id: str | None = None
    user: str | None = None


IMAGE_SIZE_RE = re.compile(r"^(\d{2,5})\s*[*x\u00d7,]\s*(\d{2,5})$", re.IGNORECASE)
MIN_IMAGE_DIM = 16
MAX_IMAGE_DIM = 8192


def _parse_image_size(size: str | None) -> tuple[int, int] | None:
    if size is None or not size.strip():
        return None
    match = IMAGE_SIZE_RE.fullmatch(size.strip())
    if match is None:
        raise HTTPException(400, f"invalid size {size!r}: expected WIDTHxHEIGHT (e.g. 1152x2048 or 1152*2048)")
    width, height = int(match.group(1)), int(match.group(2))
    if not (MIN_IMAGE_DIM <= width <= MAX_IMAGE_DIM and MIN_IMAGE_DIM <= height <= MAX_IMAGE_DIM):
        raise HTTPException(400, f"size out of range: both dimensions must be within {MIN_IMAGE_DIM}..{MAX_IMAGE_DIM}")
    return width, height


def _resize_image_bytes(content: bytes, dims: tuple[int, int] | None) -> bytes:
    if dims is None:
        return content
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(content)) as img:
            fmt = img.format or "PNG"
            resized = img.resize(dims, Image.Resampling.LANCZOS)
            if fmt.upper() == "JPEG" and resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            buffer = BytesIO()
            resized.save(buffer, format=fmt)
            return buffer.getvalue()
    except Exception as exc:
        log.warning("image resize to %s failed, returning original: %s", dims, exc)
        return content


def _token_stable_id(token: str) -> str:
    return hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts: list[DeepSeekAccount] = []
    qwen_accounts: list[QwenAccount] = []
    cache_enabled = settings.cache_enabled
    deepseek_session_store = JsonStore("deepseek-sessions", "default" if cache_enabled else None)
    qwen_session_store = JsonStore("qwen-sessions", "default" if cache_enabled else None)
    deepseek_context_store = JsonStore("deepseek-contexts", "default" if cache_enabled else None)
    qwen_context_store = JsonStore("qwen-contexts", "default" if cache_enabled else None)
    deepseek_affinity_store = JsonStore("deepseek-affinities", "default" if cache_enabled else None)
    qwen_affinity_store = JsonStore("qwen-affinities", "default" if cache_enabled else None)
    if settings.usage_enabled:
        app.state.usage = init_tracker(store=JsonStore("usage", "default"), max_records=settings.usage_max_records)
    else:
        app.state.usage = None
    try:
        if settings.deepseek_tokens:
            for i, token in enumerate(settings.deepseek_tokens):
                ds_client = DeepSeekClient(token=token, timeout=settings.timeout)
                if not await ds_client.check_auth():
                    log.warning("deepseek token #%d invalid/expired, skipping", i)
                    await ds_client.aclose()
                    continue
                accounts.append(
                    DeepSeekAccount(
                        len(accounts),
                        ds_client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=deepseek_session_store,
                        stable_id=_token_stable_id(token),
                        model_config_ttl=settings.model_config_ttl,
                    )
                )
            log.info("deepseek accounts ready: %d", len(accounts))
        if settings.qwen_tokens:
            for i, token in enumerate(settings.qwen_tokens):
                qw_client = QwenClient(token=token, timeout=settings.timeout)
                if not await qw_client.check_auth():
                    log.warning("qwen token #%d invalid/expired, skipping", i)
                    await qw_client.aclose()
                    continue
                qwen_accounts.append(
                    QwenAccount(
                        len(qwen_accounts),
                        qw_client,
                        session_cache_size=settings.session_cache_size,
                        ttl=settings.session_ttl,
                        store=qwen_session_store,
                        stable_id=_token_stable_id(token),
                    )
                )
            log.info("qwen accounts ready: %d", len(qwen_accounts))
        if accounts:
            app.state.pool = AccountPool(
                accounts,
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=deepseek_context_store,
                affinity_store=deepseek_affinity_store,
            )
        else:
            app.state.pool = None
        await _deepseek_model_configs(app.state.pool)
        if qwen_accounts:
            app.state.qwen_pool = AccountPool(
                qwen_accounts,
                label="qwen",
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=qwen_context_store,
                affinity_store=qwen_affinity_store,
            )
            app.state.qwen_models = await _fetch_qwen_models(qwen_accounts[0].client)
        else:
            app.state.qwen_pool = None
            app.state.qwen_models = []
        if not accounts and not qwen_accounts:
            raise RuntimeError("no valid credentials: set DEEPSEEK_TOKENS or QWEN_TOKENS")
        yield
    finally:
        for ds_acct in accounts:
            await ds_acct.client.aclose()
        for qw_acct in qwen_accounts:
            await qw_acct.client.aclose()


async def _fetch_qwen_models(client: QwenClient) -> list[dict]:
    try:
        raw = await client.fetch_models()
    except QwenError as exc:
        log.warning("qwen models fetch failed, using defaults: %s", exc)
        return QWEN_DEFAULT_MODELS
    models = []
    for model in raw:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        info = model.get("info")
        meta = (info or {}).get("meta") if isinstance(info, dict) else None
        chat_types = (meta or {}).get("chat_type") if isinstance(meta, dict) else None
        chat_types = chat_types or []
        if "t2t" in chat_types:
            model_type = "chat"
        elif "t2i" in chat_types:
            model_type = "image"
        elif "t2v" in chat_types:
            model_type = "video"
        elif chat_types:
            model_type = "chat"
        else:
            model_type = "chat"
        models.append(
            {
                "id": model["id"],
                "name": model.get("name") or model["id"],
                "owned_by": "qwen",
                "model_type": model_type,
                "chat_types": chat_types,
            }
        )
    if not models:
        log.warning("qwen models fetch returned nothing, using defaults")
        return QWEN_DEFAULT_MODELS
    return models


app = FastAPI(
    title="FreeTokenAPI", version=__version__, lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None,
)
chat_router = APIRouter(route_class=ChatRoute, tags=["Chat Completions"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/v1", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/v1/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    # A reachability probe must not depend on redirects, docs assets, or tokens.
    return {
        "service": "FreeTokenAPI",
        "version": __version__,
        "status": "ok",
        "api": "/v1",
        "adapter_features": [
            "responses_namespace_tools", "messages_inline_instructions", "chat_completions_developer",
            "qwen_chat_attachments", "native_web_search", "agent_tool_continuation", "deepseek_model_configs",
        ],
    }


@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(status_code=204)


def _env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def _read_env_tokens() -> tuple[list[str], list[str]]:
    env_file = _env_path()
    if not env_file.exists():
        return [], []
    ds_tokens = ""
    qw_tokens = ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("DEEPSEEK_TOKENS="):
            ds_tokens = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("QWEN_TOKENS="):
            qw_tokens = stripped.split("=", 1)[1].strip()
    ds_list = [t.strip() for t in ds_tokens.split(",") if t.strip()] if ds_tokens else []
    qw_list = [t.strip() for t in qw_tokens.split(",") if t.strip()] if qw_tokens else []
    return ds_list, qw_list


def _write_env_tokens(ds_tokens: list[str], qw_tokens: list[str]) -> None:
    env_file = _env_path()
    ds_line = f"DEEPSEEK_TOKENS={','.join(ds_tokens)}"
    qw_line = f"QWEN_TOKENS={','.join(qw_tokens)}"
    new_lines: list[str] = []
    ds_set = False
    qw_set = False
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("DEEPSEEK_TOKENS="):
                new_lines.append(ds_line)
                ds_set = True
            elif stripped.startswith("QWEN_TOKENS="):
                new_lines.append(qw_line)
                qw_set = True
            else:
                new_lines.append(line)
    if not ds_set:
        new_lines.append(ds_line)
    if not qw_set:
        new_lines.append(qw_line)
    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@app.post("/v1/tokens")
async def add_tokens(tokens: dict) -> dict:
    new_ds = [t.strip() for t in tokens.get("deepseek_tokens", []) if t.strip()]
    new_qw = [t.strip() for t in tokens.get("qwen_tokens", []) if t.strip()]
    if not new_ds and not new_qw:
        raise HTTPException(400, "no tokens provided")

    existing_ds, existing_qw = _read_env_tokens()
    ds_to_add = [t for t in new_ds if t not in existing_ds]
    qw_to_add = [t for t in new_qw if t not in existing_qw]

    if not ds_to_add and not qw_to_add:
        raise HTTPException(400, "all provided tokens already exist")

    added_ds = 0
    added_qw = 0
    skipped_ds = 0
    skipped_qw = 0

    cache_enabled = settings.cache_enabled
    ds_store = JsonStore("deepseek-sessions", "default" if cache_enabled else None)
    qw_store = JsonStore("qwen-sessions", "default" if cache_enabled else None)
    ds_context_store = JsonStore("deepseek-contexts", "default" if cache_enabled else None)
    qw_context_store = JsonStore("qwen-contexts", "default" if cache_enabled else None)
    ds_affinity_store = JsonStore("deepseek-affinities", "default" if cache_enabled else None)
    qw_affinity_store = JsonStore("qwen-affinities", "default" if cache_enabled else None)

    pool: AccountPool | None = getattr(app.state, "pool", None)
    qwen_pool: AccountPool | None = getattr(app.state, "qwen_pool", None)

    for token in ds_to_add:
        client = DeepSeekClient(token=token, timeout=settings.timeout)
        if not await client.check_auth():
            log.warning("new deepseek token invalid/expired, skipping")
            await client.aclose()
            skipped_ds += 1
            continue
        acct = DeepSeekAccount(
            len(pool.accounts) if pool else 0,
            client,
            session_cache_size=settings.session_cache_size,
            ttl=settings.session_ttl,
            store=ds_store,
            stable_id=_token_stable_id(token),
            model_config_ttl=settings.model_config_ttl,
        )
        if pool is None:
            pool = AccountPool(
                [acct],
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=ds_context_store,
                affinity_store=ds_affinity_store,
            )
            app.state.pool = pool
        else:
            pool.add_account(acct)
        try:
            await acct.model_config.get()
        except DeepSeekError as exc:
            _handle_account_error(acct, exc)
        added_ds += 1
        log.info("hot-added deepseek token (total accounts: %d)", len(pool.accounts))

    for token in qw_to_add:
        qw_client = QwenClient(token=token, timeout=settings.timeout)
        if not await qw_client.check_auth():
            log.warning("new qwen token invalid/expired, skipping")
            await qw_client.aclose()
            skipped_qw += 1
            continue
        qw_acct = QwenAccount(
            len(qwen_pool.accounts) if qwen_pool else 0,
            qw_client,
            session_cache_size=settings.session_cache_size,
            ttl=settings.session_ttl,
            store=qw_store,
            stable_id=_token_stable_id(token),
        )
        if qwen_pool is None:
            qwen_pool = AccountPool(
                [qw_acct],
                label="qwen",
                session_cache_size=settings.session_cache_size,
                ttl=settings.session_ttl,
                context_store=qw_context_store,
                affinity_store=qw_affinity_store,
            )
            app.state.qwen_pool = qwen_pool
            try:
                app.state.qwen_models = await _fetch_qwen_models(qw_client)
            except Exception as exc:
                log.warning("failed to prefetch qwen models: %s", exc)
        else:
            qwen_pool.add_account(qw_acct)
        added_qw += 1
        log.info("hot-added qwen token (total accounts: %d)", len(qwen_pool.accounts))

    merged_ds = existing_ds + ds_to_add
    merged_qw = existing_qw + qw_to_add
    _write_env_tokens(merged_ds, merged_qw)
    settings.deepseek_tokens = merged_ds
    settings.qwen_tokens = merged_qw

    parts = []
    if added_ds:
        parts.append(f"deepseek: +{added_ds}")
    if added_qw:
        parts.append(f"qwen: +{added_qw}")
    if skipped_ds:
        parts.append(f"deepseek skipped: {skipped_ds}")
    if skipped_qw:
        parts.append(f"qwen skipped: {skipped_qw}")

    return {
        "success": True,
        "message": "Tokens added and activated." if (added_ds or added_qw) else "No valid tokens to add.",
        "added": {"deepseek": added_ds, "qwen": added_qw},
        "skipped": {"deepseek": skipped_ds, "qwen": skipped_qw},
    }


async def _extract_request_model(request: Request) -> str | None:
    try:
        body = await request.body()
    except Exception:
        return None
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        model = payload.get("model")
        if isinstance(model, str):
            return model
    return None


def _log_request_failure(request: Request, model: str | None, duration: float, status: int | None = None, exc: Exception | None = None) -> None:
    model_part = f"model={model}" if model else "model=?"
    if status is not None:
        reason = f"status={status}"
    else:
        reason = f"error={str(exc) if exc else 'unknown'}"
    log.warning(
        "%s %s failed: %s %s (%.0fms)",
        request.method,
        request.url.path,
        model_part,
        reason.replace("{", "{{").replace("}", "}}"),
        duration,
    )


def _log_request_success(request: Request, duration: float) -> None:
    log.info(
        "%s %s success (%.0fms)",
        request.method,
        request.url.path,
        duration,
    )


@app.middleware("http")
async def _log_request_failures(request: Request, call_next):
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:
        _log_request_failure(
            request,
            await _extract_request_model(request),
            (time.monotonic() - started) * 1000,
            exc=exc,
        )
        raise
    if response.status_code >= 400:
        _log_request_failure(
            request,
            await _extract_request_model(request),
            (time.monotonic() - started) * 1000,
            status=response.status_code,
        )
    elif request.url.path in {"/v1/chat/completions", "/v1/responses", "/v1/messages"}:
        _log_request_success(request, (time.monotonic() - started) * 1000)
    return response


MAX_FILES_PER_REQUEST = 50
MAX_FILE_SIZE = 100 * 1024 * 1024


@dataclass
class Attachment:
    data: bytes
    name: str
    content_type: str
    is_image: bool


def _split_data_uri(uri: str) -> tuple[str, bytes]:
    if not uri.startswith("data:"):
        raise HTTPException(400, "image_url must be a data URI (data:<mime>;base64,...)")
    meta, _, payload = uri[5:].partition(",")
    if not payload:
        raise HTTPException(400, "invalid data URI: missing base64 payload")
    if not meta.endswith(";base64"):
        raise HTTPException(400, "attachment data URI must use base64 encoding")
    content_type = meta.split(";", 1)[0] or "application/octet-stream"
    if not re.fullmatch(r"[a-zA-Z0-9!#$&^_.+-]+/[a-zA-Z0-9!#$&^_.+-]+", content_type):
        raise HTTPException(400, "invalid attachment content type")
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise HTTPException(400, "invalid base64 in attachment") from exc
    return content_type, data


def _file_attachment(name: Any, content: Any, content_type: str | None = None) -> Attachment:
    if not isinstance(name, str) or not name.strip() or not isinstance(content, str) or not content:
        raise HTTPException(400, "each file needs name and base64 content")
    # Names are metadata only, never paths to read from this machine.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in {".", ".."} or any(c in name for c in "\r\n\x00"):
        raise HTTPException(400, "invalid attachment filename")
    if content.startswith("data:"):
        mime, data = _split_data_uri(content)
    else:
        mime = content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
        if not re.fullmatch(r"[a-zA-Z0-9!#$&^_.+-]+/[a-zA-Z0-9!#$&^_.+-]+", mime):
            raise HTTPException(400, "invalid attachment content type")
        try:
            data = base64.b64decode(content, validate=True)
        except ValueError as exc:
            raise HTTPException(400, "invalid base64 in attachment") from exc
    return Attachment(data, name, mime, mime.startswith("image/"))


def _collect_attachments(req: ChatCompletionRequest) -> list[Attachment]:
    attachments: list[Attachment] = []
    for msg in req.messages:
        if not isinstance(msg.content, list):
            continue
        for item in msg.content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, str):
                    uri = image_url
                elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    uri = image_url["url"]
                else:
                    raise HTTPException(400, "invalid image_url value")
                content_type, data = _split_data_uri(uri)
                if not content_type.startswith("image/"):
                    raise HTTPException(400, "image_url must contain image data")
                extension = mimetypes.guess_extension(content_type) or ".bin"
                attachments.append(Attachment(data, f"image_{len(attachments)}{extension}", content_type, True))
            elif item.get("type") in ("file", "input_file"):
                file = item.get("file") if item.get("type") == "file" else item
                if not isinstance(file, dict) or file.get("file_id") or file.get("file_url"):
                    raise HTTPException(400, "use inline file_data and filename; remote files and file IDs are not supported")
                attachments.append(_file_attachment(file.get("filename", "attachment"), file.get("file_data")))
    for f in req.files or []:
        attachments.append(_file_attachment(f.name, f.content, f.content_type))
    return attachments


def _validate_attachments(attachments: list[Attachment]) -> None:
    if not attachments:
        return
    if len(attachments) > MAX_FILES_PER_REQUEST:
        raise HTTPException(400, f"too many files: max {MAX_FILES_PER_REQUEST} per request")
    for att in attachments:
        if not att.data:
            raise HTTPException(400, "empty attachments are not supported")
        if len(att.data) > MAX_FILE_SIZE:
            raise HTTPException(400, f"file {att.name} exceeds 100 MB limit")


async def _fresh_pow_upload_headers(account) -> dict:
    try:
        return await account.pow_upload.make_header(lambda: account.client.create_pow_challenge("/api/v0/file/upload_file"))
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc


async def _upload_attachments(account, attachments: list[Attachment], model_type: str, thinking: bool) -> list[str]:
    file_ids: list[str] = []
    for att in attachments:
        for attempt in range(2):
            pow_headers = await _fresh_pow_upload_headers(account)
            try:
                info = await account.client.upload_file(
                    att.data, att.name, att.content_type, model_type,
                    thinking_enabled=thinking, pow_headers=pow_headers,
                )
                break
            except DeepSeekError as exc:
                if exc.biz_code == 40301 and attempt == 0:
                    log.info("deepseek upload proof rejected; generating a new proof and retrying once")
                    continue
                _handle_account_error(account, exc)
                raise HTTPException(_deepseek_status(exc), f"file upload failed: {exc}") from exc
        try:
            if info.get("status") is not None:
                info = await account.client.wait_for_file(info, timeout=settings.timeout)
        except DeepSeekError as exc:
            _handle_account_error(account, exc)
            status = exc.biz_code if exc.biz_code in (400, 504) else _deepseek_status(exc)
            raise HTTPException(status, f"file parsing failed: {exc}") from exc
        file_id = info.get("id")
        if not file_id:
            raise HTTPException(502, f"file upload failed for {att.name}: no file id")
        file_ids.append(file_id)
    return file_ids


async def _upload_qwen_attachments(account, attachments: list[Attachment]) -> list[dict]:
    files = []
    for attachment in attachments:
        try:
            files.append(await account.client.upload_file(attachment.data, attachment.name, attachment.content_type))
        except QwenError as exc:
            qwen_api._handle_account_error(account, exc)
            status = exc.code if isinstance(exc.code, int) and exc.code in {400, 413, 415, 504} else qwen_api._error_status(exc.code)
            raise HTTPException(status, f"Qwen attachment upload failed: {exc.message}") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Qwen attachment upload failed: upstream network error") from exc
    return files


def _resolve_model(model: str) -> str:
    if model not in MODEL_ALIASES:
        raise HTTPException(404, f"Unknown model: {model}. Use deepseek-web or deepseek-web-thinking for DeepSeek.")
    return MODEL_TYPE


def _is_reasoning_model(model: str) -> bool:
    return MODEL_ALIASES.get(model, False)


def _finish_reason(status: Any) -> str:
    if isinstance(status, str):
        return STATUS_TO_FINISH_REASON.get(status, "stop")
    return "stop"


def _pool_stats(pool) -> dict | None:
    if pool is None:
        return None
    try:
        return pool.stats()
    except Exception:
        return None


def _usage_summary() -> dict | None:
    tracker = getattr(app.state, "usage", None)
    if tracker is None:
        return None
    try:
        return tracker.snapshot()["totals"]
    except Exception:
        return None


@app.get("/health")
async def health() -> dict:
    pool = getattr(app.state, "pool", None)
    qwen_pool = getattr(app.state, "qwen_pool", None)
    return {
        "status": "ok",
        "deepseek": pool is not None,
        "qwen": qwen_pool is not None,
        "deepseek_stats": _pool_stats(pool),
        "qwen_stats": _pool_stats(qwen_pool),
        "usage": _usage_summary(),
    }


@app.get("/v1/usage")
async def usage_stats() -> dict:
    tracker = getattr(app.state, "usage", None)
    if tracker is None:
        raise HTTPException(404, "usage tracking is disabled")
    return tracker.snapshot()


async def _deepseek_model_configs(pool: AccountPool | None) -> tuple[dict[int, WebModel], list[DeepSeekError]]:
    if pool is None:
        return {}, []
    accounts = list(pool.healthy)
    results = await asyncio.gather(*(account.model_config.get() for account in accounts), return_exceptions=True)
    models: dict[int, WebModel] = {}
    errors: list[DeepSeekError] = []
    for account, result in zip(accounts, results):
        if isinstance(result, DeepSeekError):
            _handle_account_error(account, result)
            errors.append(result)
        elif isinstance(result, BaseException):
            raise result
        elif result is not None and not account.broken:
            models[account.index] = result
    return models, errors


def _model_config_failure(errors: list[DeepSeekError]) -> HTTPException:
    status = 504 if errors and all(error.biz_code == 504 for error in errors) else 502
    return HTTPException(status, "DeepSeek model configuration is unavailable; retry after the refresh cooldown")


@app.get("/v1/models")
async def list_models() -> dict:
    catalog, errors = await _deepseek_model_configs(getattr(app.state, "pool", None))
    models = advertised_models(list(catalog.values()))
    qwen_models: list[dict] = getattr(app.state, "qwen_models", [])
    for model in qwen_models:
        models.append(
            {
                "id": model["id"],
                "object": "model",
                "created": 0,
                "owned_by": model.get("owned_by", "qwen"),
                "name": model.get("name"),
                "model_type": model.get("model_type", "chat"),
            }
        )
    if not models and errors:
        raise _model_config_failure(errors)
    return {"object": "list", "data": models}


RETRYABLE_FINISH_REASONS = {
    "expert_busy_use_default",
    "parallel_chat_limit",
    "server_busy",
    "busy",
}
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 1.0
RETRY_BACKOFF_MAX_SEC = 8.0

DEEPSEEK_AUTH_ERROR_CODES = {40001, 40002, 40003, 40012, 40029}


async def _human_delay() -> None:
    delay = random.uniform(settings.human_delay_min, settings.human_delay_max)
    if delay > 0:
        await asyncio.sleep(delay)


def _resolve_provider(model: str) -> str:
    if model.startswith("qwen"):
        return "qwen"
    if model in MODEL_ALIASES:
        return "deepseek"
    if model.startswith("deepseek"):
        _resolve_model(model)
    qwen_models = getattr(app.state, "qwen_models", [])
    for m in qwen_models:
        if m.get("id") == model:
            return "qwen"
    raise HTTPException(404, f"Unknown model: {model}")


@chat_router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    provider = _resolve_provider(req.model)
    if provider == "qwen":
        return await _chat_completions_qwen(req)
    return await _chat_completions_deepseek(req)


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest) -> dict:
    pool: AccountPool = app.state.qwen_pool
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured (required for image generation)")

    dims = _parse_image_size(req.size)

    account, existing_sid = await _acquire_account(pool, req.session_id)

    try:
        result = await qwen_api.collect_image(
            account=account,
            pool=pool,
            existing_sid=existing_sid,
            lock=account.sem,
            prompt=req.prompt,
            model=req.model,
            model_id=req.model,
            user=req.user,
        )
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None

    want_b64 = req.response_format == "b64_json"
    data: list[dict] = []
    for url in result["image_urls"]:
        if not (want_b64 or dims):
            data.append({"url": url})
            continue
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as hc:
                img_resp = await hc.get(url)
            if img_resp.status_code != 200:
                log.warning("image download failed (%s) for %s, returning url", img_resp.status_code, url)
                data.append({"url": url})
                continue
            payload_bytes = _resize_image_bytes(img_resp.content, dims)
            data.append({"b64_json": base64.b64encode(payload_bytes).decode()})
        except Exception as exc:
            log.warning("image fetch failed for %s, returning url: %s", url, exc)
            data.append({"url": url})

    if not data:
        data.append({"url": "", "revised_prompt": result.get("revised_prompt", "")})

    return {
        "created": int(time.time()),
        "data": data,
        "usage": result.get("usage"),
        "session_id": result.get("session_id"),
    }


async def _acquire_account(pool: AccountPool, session_id: str | None, allowed_indices: set[int] | None = None):
    try:
        if allowed_indices is not None:
            return await pool.acquire(session_id, settings.acquire_timeout, allowed_indices=allowed_indices)
        return await pool.acquire(session_id, settings.acquire_timeout)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


def _can_reuse_session(account: Any, session_id: str | None, **kwargs: Any) -> bool:
    return bool(account.sessions.can_reuse(session_id, **kwargs))


async def _acquire_and_build(
    pool: AccountPool,
    req: ChatCompletionRequest,
    reuse_kwargs: dict[str, Any] | None = None,
    allowed_indices: set[int] | None = None,
) -> tuple[Any, str | None, tuple[str, ...], str, bool]:
    selection = {"allowed_indices": allowed_indices} if allowed_indices is not None else {}
    context_seq = toolemu.context_sequence(req.messages, user=getattr(req, "user", None))
    interleaved = toolemu.has_interleaved_system(req.messages)
    if interleaved:
        # A later instruction may change how earlier turns should be read.
        # Do not append a partial/duplicated transcript to a cached web session.
        account, _ = await _acquire_account(pool, None, **selection)
        existing_sid = None
    elif req.session_id:
        owner = pool.account_for_session(req.session_id) if allowed_indices is not None else None
        account, existing_sid = await _acquire_account(pool, req.session_id, **selection)
        if existing_sid is None and (allowed_indices is None or owner is None or owner is account):
            # Preserve client-chosen session keys, but never reuse a different
            # account's session when capability routing requires a new account.
            existing_sid = req.session_id
    else:
        cached_sid = pool.resolve_context(context_seq) if context_seq else None
        account, existing_sid = await _acquire_account(pool, cached_sid, **selection)
    has_session = not interleaved and _can_reuse_session(account, existing_sid, **(reuse_kwargs or {}))
    try:
        prompt, tool_mode = toolemu.build_prompt(
            req.messages,
            getattr(req, "tools", None),
            getattr(req, "tool_choice", None),
            has_session,
            getattr(req, "response_format", None),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return account, existing_sid, context_seq, prompt, tool_mode


def _include_usage(req: ChatCompletionRequest) -> bool:
    opts = getattr(req, "stream_options", None)
    if not isinstance(opts, dict):
        return False
    return bool(opts.get("include_usage"))


def _deepseek_usage(total: int, prompt: str = "") -> dict:
    value = max(0, int(total or 0))
    prompt_tokens = estimate_tokens(prompt)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": value, "total_tokens": prompt_tokens + value}


def _advance_session_usage(session, accumulated_total: int) -> int:
    prev = max(0, int(getattr(session, "accumulated_tokens", 0) or 0))
    current = max(0, int(accumulated_total or 0))
    session.accumulated_tokens = max(prev, current)
    return max(0, current - prev)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error_sse(
    chunk_id: str,
    created: int,
    model: str,
    message: str,
    session_key: str | None = None,
    error_finish: str | None = None,
    choice_finish: str | None = None,
) -> tuple[str, str, str]:
    error: dict = {"message": message}
    if error_finish is not None:
        error["finish_reason"] = error_finish
    error_chunk = _sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "error": error,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice_finish or error_finish or "error"}],
        }
    )
    tail_chunk = _sse(
        {
            "id": chunk_id,
            "session_id": session_key,
            "object": "chat.completion.chunk",
            "choices": [],
        }
    )
    return error_chunk, tail_chunk, "data: [DONE]\n\n"


async def _stream_guard(gen, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        async for item in gen:
            yield item
    except AccountPoolBusy:
        for line in _stream_error_sse(chunk_id, created, model, "all accounts are busy, try again later"):
            yield line
    except Exception as exc:
        log.exception("stream generator failed: %s", exc)
        msg = str(exc) or repr(exc) or "unknown stream error"
        for line in _stream_error_sse(chunk_id, created, model, f"stream error: {msg}"):
            yield line
    finally:
        await close_iterator(gen)


async def _chat_completions_deepseek(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.pool
    if pool is None:
        raise HTTPException(503, "deepseek provider is not configured")

    model_type = _resolve_model(req.model)
    thinking = req.thinking if req.thinking is not None else _is_reasoning_model(req.model)
    search = settings.search_enabled if req.search is None else req.search

    attachments = _collect_attachments(req)
    _validate_attachments(attachments)
    catalog, errors = await _deepseek_model_configs(pool)
    if not catalog:
        if errors:
            raise _model_config_failure(errors)
        raise HTTPException(503, "No account has an enabled, switchable DeepSeek default web model")
    rejected = {index: model.rejection(thinking=thinking, search=search, attachments=attachments) for index, model in catalog.items()}
    eligible = {index for index, reason in rejected.items() if reason is None}
    if not eligible:
        # An account whose configuration failed may support the requested
        # capability. Do not incorrectly turn an upstream outage into a 400.
        if errors:
            raise _model_config_failure(errors)
        detail = (
            next(iter(rejected.values())) if len(set(rejected.values())) == 1
            else "No single DeepSeek account supports this combination of thinking, search, and attachments"
        )
        raise HTTPException(400, detail)
    account, existing_sid, context_seq, prompt, tool_mode = await _acquire_and_build(pool, req, allowed_indices=eligible)
    ref_file_ids = None
    if attachments:
        # Keep a proof and its upload together on the account, just like chat
        # requests. Concurrent uploads must not invalidate each other's proofs.
        try:
            async with account_lock(account.sem, settings.acquire_timeout):
                ref_file_ids = await _upload_attachments(account, attachments, model_type, thinking)
        except AccountPoolBusy:
            raise HTTPException(429, "all accounts are busy, try again later") from None

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_type": model_type,
        "thinking": thinking,
        "search": search,
        "ref_file_ids": ref_file_ids,
        "tool_schemas": toolemu.tool_schema_map(getattr(req, "tools", None)),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
        "reduced_prompts": _reduced_prompt_variants(
            req.messages, getattr(req, "tools", None), getattr(req, "tool_choice", None), getattr(req, "response_format", None), prompt
        ),
        "messages": req.messages,
        "tools": getattr(req, "tools", None),
        "tool_choice": getattr(req, "tool_choice", None),
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
    }
    if req.stream:
        return StreamingResponse(
            _stream_guard(_stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await _collect_non_stream(lock=account.sem, **common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _chat_completions_qwen(req: ChatCompletionRequest) -> Any:
    pool: AccountPool = app.state.qwen_pool
    if pool is None:
        raise HTTPException(503, "qwen provider is not configured")

    thinking = req.thinking if req.thinking is not None else True
    search = settings.search_enabled if req.search is None else req.search

    attachments = _collect_attachments(req)
    _validate_attachments(attachments)
    account, existing_sid, context_seq, prompt, tool_mode = await _acquire_and_build(pool, req, {"model": req.model})
    files = await _upload_qwen_attachments(account, attachments) if attachments else None

    common = {
        "account": account,
        "pool": pool,
        "existing_sid": existing_sid,
        "prompt": prompt,
        "model": req.model,
        "model_id": req.model,
        "thinking": thinking,
        "search": search,
        "tool_schemas": toolemu.tool_schema_map(getattr(req, "tools", None)),
        "tool_mode": tool_mode,
        "include_usage": _include_usage(req),
        "context_seq": context_seq,
        "messages": req.messages,
        "tools": getattr(req, "tools", None),
        "tool_choice": getattr(req, "tool_choice", None),
        "response_format": getattr(req, "response_format", None),
        "user": getattr(req, "user", None),
    }
    if files:
        common["files"] = files
    if req.stream:
        return StreamingResponse(
            _stream_guard(qwen_api.stream_openai(lock=account.sem, **common), req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        return await qwen_api.collect_non_stream(lock=account.sem, **common)
    except AccountPoolBusy:
        raise HTTPException(429, "all accounts are busy, try again later") from None


async def _prepare_session(
    account: DeepSeekAccount,
    pool: AccountPool,
    existing_sid: str | None,
    context_seq: tuple[str, ...] | None = None,
) -> tuple[DeepSeekSession, str, str | None]:
    try:
        session, session_key = await account.sessions.obtain(existing_sid)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc
    pool.register(account.index, session_key)
    if existing_sid and session_key != existing_sid:
        pool.forget(existing_sid)
        pool.forget_context(existing_sid)
        account.sessions.forget(existing_sid)
    if context_seq:
        pool.index_context(session_key, context_seq)
    return session, session_key, session.last_message_id


async def _send_completion(
    client,
    pow_headers,
    session_id,
    parent_message_id,
    prompt,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
):
    try:
        resp = await client.completion(
            chat_session_id=session_id,
            prompt=prompt,
            parent_message_id=parent_message_id,
            model_type=model_type,
            thinking_enabled=thinking,
            search_enabled=search,
            ref_file_ids=ref_file_ids,
            pow_headers=pow_headers,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text[:500]) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"DeepSeek request failed: {exc}") from exc

    if resp.status_code != 200:
        body = await resp.aread()
        await resp.aclose()
        raise HTTPException(resp.status_code, body[:500].decode("utf-8", errors="replace"))

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = await resp.aread()
        await resp.aclose()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(502, body[:500].decode("utf-8", errors="replace")) from exc
        data = payload.get("data") or {}
        if data.get("biz_code"):
            code = data["biz_code"]
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            raise HTTPException(status, f"DeepSeek error {code}: {data.get('biz_msg')}")
        if payload.get("code"):
            code = payload["code"]
            status = 401 if code in DEEPSEEK_AUTH_ERROR_CODES else 502
            raise HTTPException(
                status,
                f"DeepSeek error {code}: {payload.get('msg') or payload.get('message')}",
            )
        raise HTTPException(502, "unexpected non-stream response")
    return resp


def _is_retryable_hint(rec: MessageReconstructor) -> bool:
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") in RETRYABLE_FINISH_REASONS)


RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
STALE_SESSION_STATUSES = {400, 404}


def _retry_delay(attempt: int) -> float:
    return min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SEC)


def _is_retryable_http(exc: HTTPException) -> bool:
    return exc.status_code in RETRYABLE_HTTP_STATUSES


def _is_context_limit(rec: MessageReconstructor) -> bool:
    if rec.status == CONTEXT_LENGTH_STATUS:
        return True
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") == CONTEXT_LENGTH_STATUS)


def _is_input_exceeds_limit(rec: MessageReconstructor) -> bool:
    if rec.status == INPUT_EXCEEDS_LIMIT:
        return True
    hint = rec.hint_error
    return bool(hint and hint.get("finish_reason") == INPUT_EXCEEDS_LIMIT)


def _incomplete_message(rec: MessageReconstructor) -> str:
    hint = rec.hint_error or {}
    message = hint.get("message")
    return message if isinstance(message, str) and message else RESPONSE_INCOMPLETE_MESSAGE




def _interrupted_response_message(rec: MessageReconstructor) -> str | None:
    if rec.status in ("WIP", "INCOMPLETE", "TIMEOUT"):
        return f"DeepSeek response ended before completion ({rec.status}); retry the request"
    if rec.hint_error and not (_is_context_limit(rec) or _is_input_exceeds_limit(rec)):
        return rec.hint_error.get("message") or "DeepSeek reported an error after partial output; retry the request"
    return None

def _incomplete_error_body(message: str) -> str:
    return json.dumps({"error": {"message": message, "finish_reason": RESPONSE_INCOMPLETE}}, ensure_ascii=False)


def _input_exceeds_hint_from_http(exc: HTTPException) -> dict | None:
    detail = exc.detail
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            return None
    if not isinstance(detail, dict):
        return None
    if detail.get("finish_reason") != INPUT_EXCEEDS_LIMIT:
        return None
    message = detail.get("message")
    return {
        "message": message if isinstance(message, str) else "Content is too long",
        "finish_reason": INPUT_EXCEEDS_LIMIT,
    }


def _drop_session(pool, account, session_key) -> None:
    pool.forget(session_key)
    pool.forget_context(session_key)
    account.sessions.forget(session_key)


def _deepseek_status(exc: DeepSeekError) -> int:
    return 401 if exc.biz_code in DEEPSEEK_AUTH_ERROR_CODES else 502


async def _send_with_auth(account, *args, **kwargs):
    try:
        return await _send_completion(*args, **kwargs)
    except HTTPException as exc:
        if exc.status_code in (401, 403):
            account.mark_broken()
        raise


def _deepseek_error_detail(exc: DeepSeekError) -> str:
    if exc.biz_code in DEEPSEEK_AUTH_ERROR_CODES:
        return f"DeepSeek auth error: {exc}"
    return f"DeepSeek error: {exc}"


def _handle_account_error(account: DeepSeekAccount, exc: Exception) -> None:
    code = getattr(exc, "biz_code", None)
    if code in DEEPSEEK_AUTH_ERROR_CODES:
        account.mark_broken()
        log.warning("account #%d auth error %s: %s", account.index, code, exc)
    else:
        log.warning("account #%d error: %s", account.index, exc)


async def _fresh_pow_headers(account) -> dict:
    try:
        return await account.pow.make_header(account.client.create_pow_challenge)
    except DeepSeekError as exc:
        _handle_account_error(account, exc)
        raise HTTPException(_deepseek_status(exc), _deepseek_error_detail(exc)) from exc


def _busy_error_body(rec: MessageReconstructor) -> str:
    hint = rec.hint_error or {}
    return json.dumps(
        {
            "error": {
                "message": hint.get("message") or "DeepSeek server is busy, try again later",
                "finish_reason": hint.get("finish_reason"),
            }
        },
        ensure_ascii=False,
    )


async def _try_stop_stream(client, session_id: str, message_id: str | None) -> None:
    if not session_id or not message_id:
        return
    try:
        await client.stop_stream(session_id, message_id)
    except Exception as exc:
        log.debug("stop_stream failed for %s: %s", session_id, exc)


def _build_assistant_message(
    content: str,
    reasoning: str | None,
    tool_mode: bool,
    tool_schemas: dict | None,
) -> tuple[dict, str]:
    if tool_mode:
        parsed = toolemu.parse_tool_calls(content, tool_schemas)
        if parsed is not None:
            tool_calls, tool_text = parsed
            if tool_calls:
                return toolemu.format_tool_message(tool_calls, tool_text, reasoning), "tool_calls"
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return message, "stop"


def _build_completion_response(
    model: str,
    message: dict,
    finish: str,
    usage: dict,
    session_key: str | None,
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": usage,
        "session_id": session_key,
    }


async def _send_deepseek_stream(
    account,
    session,
    parent_message_id,
    prompt,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
) -> tuple[MessageReconstructor, str | None, str | None]:
    pow_headers = await _fresh_pow_headers(account)
    resp = await _send_with_auth(
        account,
        account.client,
        pow_headers,
        session.id,
        parent_message_id,
        prompt,
        model_type,
        thinking,
        search,
        ref_file_ids,
    )
    rec = MessageReconstructor()
    incremental = IncrementalSSE()
    response_message_id: str | None = None
    stop_message_id: str | None = None
    try:
        async for chunk in resp.aiter_bytes():
            for event in incremental.feed(chunk):
                if event.event == "ready" and isinstance(event.data, dict):
                    response_message_id = event.data.get("response_message_id")
                    if response_message_id:
                        stop_message_id = response_message_id
                rec.handle(event)
        for event in incremental.finish():
            rec.handle(event)
    except (httpx.HTTPError, RuntimeError) as exc:
        if rec.id:
            stop_message_id = rec.id
        await _try_stop_stream(account.client, session.id, stop_message_id)
        raise DeepSeekStreamError(f"Stream processing failed: {exc}") from exc
    except BaseException:
        if rec.id:
            stop_message_id = rec.id
        await _try_stop_stream(account.client, session.id, stop_message_id)
        raise
    finally:
        if rec.id:
            stop_message_id = rec.id
        try:
            await resp.aclose()
        except Exception as exc:
            log.debug("response close failed: %s", exc)
            await _try_stop_stream(account.client, session.id, stop_message_id)
    return rec, response_message_id, stop_message_id


async def _collect_continuation(
    account,
    session,
    parent_message_id,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
) -> MessageReconstructor | None:
    attempt = 0
    while True:
        try:
            rec, _response_message_id, _stop_message_id = await _send_deepseek_stream(
                account,
                session,
                parent_message_id,
                CONTINUE_PROMPT,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except DeepSeekStreamError as exc:
            raise HTTPException(502, str(exc)) from exc
        except HTTPException as exc:
            if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                attempt += 1
                delay = _retry_delay(attempt)
                log.warning(
                    "deepseek continuation error (%s), retry %d/%d in %.1fs",
                    exc.status_code,
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            return None
        if not (rec.content or rec.reasoning) and _is_retryable_hint(rec) and attempt < MAX_RETRIES:
            attempt += 1
            delay = _retry_delay(attempt)
            log.warning(
                "deepseek continuation retryable hint (%s), retry %d/%d in %.1fs",
                (rec.hint_error or {}).get("finish_reason"),
                attempt,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
            continue
        return rec


def _reduced_prompt_variants(
    messages: list[Any], tools: list[Any] | None, tool_choice: Any, response_format: Any, original_prompt: str
) -> list[tuple[str, bool, dict[str, Any]]]:
    variants: list[tuple[str, bool, dict[str, Any]]] = []
    if toolemu.has_interleaved_system(messages):
        # Never "recover" by moving later instructions before the conversation
        # or dropping the user question/tool results that precede them.
        return variants
    if tools:
        try:
            prompt, tool_mode = toolemu.build_prompt(messages, None, None, False, response_format)
            if prompt != original_prompt:
                variants.append((prompt, tool_mode, {}))
        except ValueError:
            pass
    system_msgs = [msg for msg in messages if getattr(msg, "role", None) == "system"]
    last_input = None
    for msg in reversed(messages):
        if getattr(msg, "role", None) in ("user", "system"):
            last_input = msg
            break
    reduced_msgs = list(system_msgs)
    if last_input is not None and getattr(last_input, "role", None) == "user":
        reduced_msgs.append(last_input)
    if reduced_msgs:
        try:
            prompt, tool_mode = toolemu.build_prompt(reduced_msgs, None, None, False, response_format)
            if prompt != original_prompt:
                variants.append((prompt, tool_mode, {}))
        except ValueError:
            pass
    if reduced_msgs and tools:
        try:
            prompt, tool_mode = toolemu.build_prompt(reduced_msgs, tools, tool_choice, False, response_format)
            if prompt != original_prompt and not any(v[0] == prompt for v in variants):
                variants.append((prompt, tool_mode, toolemu.tool_schema_map(tools)))
        except ValueError:
            pass
    return variants


async def _collect_reduced(
    account,
    pool,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]],
    model_type,
    thinking,
    search,
    ref_file_ids=None,
):
    for prompt, _tool_mode, _tool_schemas in reduced_prompts:
        session_key = None
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, None, None)
            rec, _response_message_id, _stop_message_id = await _send_deepseek_stream(
                account,
                session,
                parent_message_id,
                prompt,
                model_type,
                thinking,
                search,
                ref_file_ids,
            )
        except (HTTPException, httpx.HTTPError, DeepSeekStreamError):
            if session_key is not None:
                _drop_session(pool, account, session_key)
            continue
        if (rec.content or rec.reasoning) and not _is_input_exceeds_limit(rec):
            return rec, session, session_key
        if session_key is not None:
            _drop_session(pool, account, session_key)
    return None


async def _collect_non_stream(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    user=None,
):
    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        if session_key != existing_sid and messages is not None:
            try:
                prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
            except ValueError:
                pass
            tool_schemas = toolemu.tool_schema_map(tools)
        stop_message_id: str | None = None
        started = time.monotonic()
        had_cached_session = bool(existing_sid) and account.sessions.get(existing_sid) is not None
        stale_rebuilt = False
        rec: MessageReconstructor | None = None
        response_message_id = None
        attempt = 0

        try:
            while True:
                try:
                    rec, response_message_id, stop_message_id = await _send_deepseek_stream(
                        account,
                        session,
                        parent_message_id,
                        prompt,
                        model_type,
                        thinking,
                        search,
                        ref_file_ids,
                    )
                except DeepSeekStreamError as exc:
                    raise HTTPException(502, str(exc)) from exc
                except HTTPException as exc:
                    input_hint = _input_exceeds_hint_from_http(exc)
                    if input_hint is not None:
                        rec = MessageReconstructor()
                        rec.hint_error = input_hint
                        break
                    if exc.status_code in STALE_SESSION_STATUSES and had_cached_session and not stale_rebuilt and messages is not None:
                        stale_rebuilt = True
                        _drop_session(pool, account, session_key)
                        try:
                            prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                            tool_schemas = toolemu.tool_schema_map(tools)
                        except (ValueError, TypeError, AttributeError) as build_exc:
                            raise exc from build_exc
                        log.warning("deepseek session %s is stale (%s), rebuilt full history into a fresh chat", session_key, exc.status_code)
                        session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                        stop_message_id = None
                        response_message_id = None
                        continue
                    if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                        attempt += 1
                        delay = _retry_delay(attempt)
                        log.warning(
                            "deepseek provider error (%s), retry %d/%d in %.1fs",
                            exc.status_code,
                            attempt,
                            MAX_RETRIES,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise
                if not (rec.content or rec.reasoning) and _is_retryable_hint(rec) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "deepseek retryable hint (%s), retry %d/%d in %.1fs",
                        (rec.hint_error or {}).get("finish_reason"),
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        except BaseException:
            await _try_stop_stream(account.client, session.id, stop_message_id)
            raise

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            raise HTTPException(400, "context length exceeded: conversation too long, start a new conversation")
        incomplete_message: str | None = None
        reduced_notice: str | None = None
        if _is_input_exceeds_limit(rec):
            incomplete_message = _incomplete_message(rec)
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if not _is_input_exceeds_limit(cont_rec):
                    incomplete_message = None
                    break
                incomplete_message = _incomplete_message(cont_rec)
                if not cont_rec.content:
                    break
            if incomplete_message is not None and not (rec.content or rec.reasoning) and reduced_prompts:
                _drop_session(pool, account, session_key)
                reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                if reduced is not None:
                    rec, session, session_key = reduced
                    response_message_id = rec.id or response_message_id
                    stop_message_id = response_message_id
                    reduced_notice = REDUCED_CONTEXT_MESSAGE
        if incomplete_message is not None and reduced_notice is None:
            log.warning("deepseek response incomplete: %s", incomplete_message)
            raise HTTPException(502, _incomplete_error_body(incomplete_message))
        content = rec.content
        reasoning = rec.reasoning
        if not (content or reasoning) and rec.hint_error:
            raise HTTPException(429, _busy_error_body(rec))
        interrupted = _interrupted_response_message(rec)
        if interrupted:
            await _try_stop_stream(account.client, session.id, rec.id or response_message_id)
            raise HTTPException(502, _incomplete_error_body(interrupted))
        request_tokens = _advance_session_usage(session, rec.accumulated_tokens)
        usage = _deepseek_usage(request_tokens, prompt)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        record_usage(
            "deepseek",
            model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            user=user,
            session_id=session_key,
        )
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)
        message, finish = _build_assistant_message(content, reasoning, tool_mode, tool_schemas)
        if finish == "stop":
            finish = _finish_reason(rec.status)
        response = _build_completion_response(model, message, finish, usage, session_key)
        if reduced_notice is not None:
            log.warning("deepseek response delivered from reduced context (%s)", model)
            response["error"] = {"message": reduced_notice, "finish_reason": RESPONSE_INCOMPLETE}
            response["choices"][0]["finish_reason"] = RESPONSE_INCOMPLETE
        return response


async def _stream_openai(
    account,
    pool,
    existing_sid,
    lock,
    prompt,
    model,
    model_type,
    thinking,
    search,
    ref_file_ids=None,
    tool_mode=False,
    tool_schemas=None,
    include_usage=False,
    context_seq: tuple[str, ...] | None = None,
    reduced_prompts: list[tuple[str, bool, dict[str, Any]]] | None = None,
    messages=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    user=None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    await _human_delay()
    async with account_lock(lock, settings.acquire_timeout):
        try:
            session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            for line in _stream_error_sse(chunk_id, created, model, detail):
                yield line
            return

        if session_key != existing_sid and messages is not None:
            try:
                prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
            except ValueError:
                pass
            tool_schemas = toolemu.tool_schema_map(tools)

        rec: MessageReconstructor | None = None
        response_message_id = None
        stop_message_id: str | None = None
        content_buf = ""
        started = time.monotonic()
        had_cached_session = bool(existing_sid) and account.sessions.get(existing_sid) is not None
        stale_rebuilt = False
        attempt = 0
        while True:
            try:
                pow_headers = await _fresh_pow_headers(account)
                resp = await _send_with_auth(
                    account,
                    account.client,
                    pow_headers,
                    session.id,
                    parent_message_id,
                    prompt,
                    model_type,
                    thinking,
                    search,
                    ref_file_ids,
                )
            except HTTPException as exc:
                input_hint = _input_exceeds_hint_from_http(exc)
                if input_hint is not None:
                    rec = MessageReconstructor()
                    rec.hint_error = input_hint
                    break
                if exc.status_code in STALE_SESSION_STATUSES and had_cached_session and not stale_rebuilt and messages is not None:
                    stale_rebuilt = True
                    _drop_session(pool, account, session_key)
                    try:
                        prompt, tool_mode = toolemu.build_prompt(messages, tools, tool_choice, False, response_format)
                        tool_schemas = toolemu.tool_schema_map(tools)
                    except ValueError:
                        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                        for line in _stream_error_sse(chunk_id, created, model, detail, session_key):
                            yield line
                        return
                    log.warning("deepseek session %s is stale (%s), rebuilt full history into a fresh chat", session_key, exc.status_code)
                    session, session_key, parent_message_id = await _prepare_session(account, pool, existing_sid, context_seq)
                    stop_message_id = None
                    response_message_id = None
                    continue
                if _is_retryable_http(exc) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = _retry_delay(attempt)
                    log.warning(
                        "deepseek provider error (%s), retry %d/%d in %.1fs",
                        exc.status_code,
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                for line in _stream_error_sse(chunk_id, created, model, detail, session_key):
                    yield line
                return
            rec = MessageReconstructor()
            incremental = IncrementalSSE()
            response_message_id = None
            got_content = False
            role_sent = False
            stopped = False
            try:
                async for chunk in resp.aiter_bytes():
                    for event in incremental.feed(chunk):
                        if event.event == "ready" and isinstance(event.data, dict):
                            response_message_id = event.data.get("response_message_id")
                            if response_message_id:
                                stop_message_id = response_message_id
                        rec.handle(event)
                        c_diff, r_diff = rec.take_diffs()
                        if not (c_diff or r_diff):
                            continue
                        got_content = True
                        if not role_sent:
                            role_sent = True
                            yield _sse(
                                {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant"},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        delta: dict = {}
                        if c_diff:
                            if tool_mode:
                                content_buf += c_diff
                            else:
                                delta["content"] = c_diff
                        if r_diff:
                            delta["reasoning_content"] = r_diff
                        if delta:
                            yield _sse(
                                {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
            except BaseException:
                stopped = True
                if rec.id:
                    stop_message_id = rec.id
                await _try_stop_stream(account.client, session.id, stop_message_id)
                raise
            finally:
                if rec.id:
                    stop_message_id = rec.id
                try:
                    await resp.aclose()
                except Exception as exc:
                    log.debug("response close failed: %s", exc)
                    if not stopped:
                        await _try_stop_stream(account.client, session.id, stop_message_id)
            if got_content:
                break
            if _is_retryable_hint(rec) and attempt < MAX_RETRIES:
                attempt += 1
                delay = _retry_delay(attempt)
                log.warning(
                    "deepseek retryable hint (%s), retry %d/%d in %.1fs",
                    (rec.hint_error or {}).get("finish_reason"),
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            break

        assert rec is not None
        if _is_context_limit(rec) and not (rec.content or rec.reasoning):
            _drop_session(pool, account, session_key)
            for line in _stream_error_sse(
                chunk_id,
                created,
                model,
                "context length exceeded: conversation too long, start a new conversation",
                session_key,
                CONTEXT_LENGTH_STATUS,
                "length",
            ):
                yield line
            return
        incomplete_message: str | None = None
        reduced_notice: str | None = None
        if _is_input_exceeds_limit(rec):
            incomplete_message = _incomplete_message(rec)
            cont_parent = rec.id or response_message_id or parent_message_id
            for _ in range(MAX_CONTINUE_ROUNDS):
                cont_rec = await _collect_continuation(account, session, cont_parent, model_type, thinking, search, ref_file_ids)
                if cont_rec is None:
                    break
                rec.extend_with(cont_rec)
                cont_parent = cont_rec.id or cont_parent
                if cont_rec.content:
                    if tool_mode:
                        content_buf += cont_rec.content
                    else:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": cont_rec.content}, "finish_reason": None}],
                            }
                        )
                if cont_rec.reasoning:
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": cont_rec.reasoning}, "finish_reason": None}],
                        }
                    )
                if not _is_input_exceeds_limit(cont_rec):
                    incomplete_message = None
                    break
                incomplete_message = _incomplete_message(cont_rec)
                if not (cont_rec.content or cont_rec.reasoning):
                    break
            if incomplete_message is not None and not (rec.content or rec.reasoning) and reduced_prompts:
                _drop_session(pool, account, session_key)
                reduced = await _collect_reduced(account, pool, reduced_prompts, model_type, thinking, search, ref_file_ids)
                if reduced is not None:
                    rec, session, session_key = reduced
                    response_message_id = rec.id or response_message_id
                    stop_message_id = response_message_id
                    reduced_notice = REDUCED_CONTEXT_MESSAGE
                    if rec.content:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": rec.content}, "finish_reason": None}],
                            }
                        )
                    if rec.reasoning:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"reasoning_content": rec.reasoning}, "finish_reason": None}],
                            }
                        )
        if incomplete_message is not None and reduced_notice is None:
            log.warning("deepseek response incomplete: %s", incomplete_message)
            for line in _stream_error_sse(chunk_id, created, model, incomplete_message, session_key, RESPONSE_INCOMPLETE, RESPONSE_INCOMPLETE):
                yield line
            return
        if not (rec.content or rec.reasoning) and rec.hint_error:
            hint = rec.hint_error
            for line in _stream_error_sse(
                chunk_id,
                created,
                model,
                hint.get("message") or "DeepSeek server is busy, try again later",
                session_key,
                hint.get("finish_reason"),
            ):
                yield line
            return

        interrupted = _interrupted_response_message(rec)
        if interrupted:
            await _try_stop_stream(account.client, session.id, rec.id or response_message_id)
            for line in _stream_error_sse(chunk_id, created, model, interrupted, session_key, RESPONSE_INCOMPLETE, RESPONSE_INCOMPLETE):
                yield line
            return

        request_tokens = _advance_session_usage(session, rec.accumulated_tokens)
        usage = _deepseek_usage(request_tokens, prompt)
        account.sessions.touch_last_message(session_key, rec.id or response_message_id)
        record_usage(
            "deepseek",
            model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            user=user,
            session_id=session_key,
        )
        log.info("deepseek completion success (%.0fms)", (time.monotonic() - started) * 1000)

        if tool_mode:
            parsed = toolemu.parse_tool_calls(content_buf or rec.content, tool_schemas)
            if parsed is not None:
                tool_calls, tool_text = parsed
                if tool_calls:
                    for delta in toolemu.tool_call_deltas(tool_calls, tool_text):
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            }
                        )
                    finish = "tool_calls"
                else:
                    if content_buf:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content_buf},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    finish = _finish_reason(rec.status)
            else:
                if content_buf:
                    yield _sse(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content_buf},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                finish = _finish_reason(rec.status)
        else:
            finish = _finish_reason(rec.status)

        if reduced_notice is not None:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": {"message": reduced_notice, "finish_reason": RESPONSE_INCOMPLETE},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": RESPONSE_INCOMPLETE}],
                }
            )
        else:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                }
            )
        if include_usage:
            yield _sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
        yield _sse(
            {
                "id": chunk_id,
                "session_id": session_key,
                "object": "chat.completion.chunk",
                "choices": [],
            }
        )
        yield "data: [DONE]\n\n"


async def _native_chat_completion(payload: dict[str, Any]) -> Any:
    """Share provider routing directly, without an HTTP round-trip to ourselves."""
    try:
        req = ChatCompletionRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(400, "Invalid normalized chat request") from exc
    if req.parallel_tool_calls is False and req.tools:
        req.messages.insert(0, ChatMessage(role="system", content="Call at most one tool per assistant response. Wait for its result before calling another tool."))
    return await chat_completions(req)


app.include_router(chat_router)
app.include_router(create_responses_router(_native_chat_completion))
app.include_router(create_messages_router(_native_chat_completion))
