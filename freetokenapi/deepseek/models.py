"""Account-scoped capabilities for DeepSeek's unified web model.

Public API names are adapter aliases, not official paid-API model identifiers.
Only the web backend's enabled, switchable default entry is exposed.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from .client import DeepSeekError

MODEL_ALIASES = {"deepseek-web": False, "deepseek-web-thinking": True}
MODEL_TYPE = "default"


class AttachmentLike(Protocol):
    data: bytes
    name: str
    is_image: bool


@dataclass(frozen=True)
class FileCapabilities:
    vision: bool
    conflict_with_search: bool
    max_count: int | None
    max_size: int | None
    extensions: frozenset[str] | None


@dataclass(frozen=True)
class WebModel:
    thinking: bool
    search: bool
    files: FileCapabilities | None
    model_type: str = MODEL_TYPE

    @property
    def supports_files(self) -> bool:
        return self.files is not None and self.files.max_count != 0 and self.files.max_size != 0

    def rejection(self, *, thinking: bool, search: bool, attachments: Sequence[AttachmentLike]) -> str | None:
        if thinking and not self.thinking:
            return "The current DeepSeek web model does not support thinking on this account"
        if search and not self.search:
            return "The current DeepSeek web model does not support native search on this account"
        if not attachments:
            return None
        features = self.files
        if features is None or not self.supports_files:
            return "The current DeepSeek web model does not support file attachments on this account"
        if search and features.conflict_with_search:
            return "The current DeepSeek web model does not allow attachments together with native search on this account"
        if features.max_count is not None and len(attachments) > features.max_count:
            return f"DeepSeek allows at most {features.max_count} attachments on this account"
        for attachment in attachments:
            if attachment.is_image and not features.vision:
                return "The current DeepSeek web model does not support image understanding on this account"
            if features.max_size is not None and len(attachment.data) > features.max_size:
                return f"DeepSeek attachment exceeds this account's {features.max_size}-byte upload limit"
            if features.extensions is not None:
                suffix = PurePosixPath(attachment.name.replace("\\", "/")).suffix.lower().lstrip(".")
                if suffix not in features.extensions:
                    return f"DeepSeek does not advertise support for the .{suffix or '(none)'} file extension on this account"
        return None


def _feature(entry: dict, name: str) -> dict | None:
    value = entry.get(name)
    if value is not None and not isinstance(value, dict):
        raise DeepSeekError(502, f"Invalid DeepSeek model configuration: {name} must be an object or null")
    return value


def _limit(feature: dict, name: str) -> int | None:
    value = feature.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeepSeekError(502, f"Invalid DeepSeek model configuration: {name}")
    return value


def _flag(entry: dict, name: str) -> bool:
    value = entry.get(name, False)
    if not isinstance(value, bool):
        raise DeepSeekError(502, f"Invalid DeepSeek model configuration: {name} must be boolean")
    return value


def parse_default_model(configs: list[dict]) -> WebModel | None:
    if not isinstance(configs, list) or any(not isinstance(item, dict) for item in configs):
        raise DeepSeekError(502, "Invalid DeepSeek model_configs: expected a list of objects")
    defaults = [item for item in configs if item.get("model_type") == MODEL_TYPE]
    if len(defaults) > 1:
        raise DeepSeekError(502, "Invalid DeepSeek model_configs: duplicate default model")
    if not defaults:
        return None
    entry = defaults[0]
    if not (_flag(entry, "enabled") and _flag(entry, "switchable")):
        return None
    files = _feature(entry, "file_feature")
    file_features = None
    if files is not None:
        extensions = files.get("support_file_exts")
        if extensions is not None and (not isinstance(extensions, list) or any(not isinstance(ext, str) or not ext.strip() for ext in extensions)):
            raise DeepSeekError(502, "Invalid DeepSeek model configuration: support_file_exts")
        file_features = FileCapabilities(
            vision=_flag(files, "vision"),
            conflict_with_search=_flag(files, "conflict_with_search"),
            max_count=_limit(files, "max_input_file_count"),
            max_size=_limit(files, "max_upload_file_size"),
            extensions=frozenset(ext.strip().lower().lstrip(".") for ext in extensions) if extensions else None,
        )
    return WebModel(thinking=_feature(entry, "think_feature") is not None, search=_feature(entry, "search_feature") is not None, files=file_features)


class ModelConfigCache:
    """TTL cache with single-flight refresh and a short error cooldown.

    An expired snapshot is never used to guess permissions after a failed
    refresh. No credentials, settings JWTs, or account data are persisted.
    """

    def __init__(self, client, ttl: float = 300.0, *, timeout: float = 10.0, clock: Callable[[], float] | None = None) -> None:
        self._client = client
        self.ttl = ttl if math.isfinite(ttl) and ttl >= 0 else 300.0
        self.timeout = timeout
        self._clock = clock or time.monotonic
        self._model: WebModel | None = None
        self._expires_at = -math.inf
        self._retry_at = 0.0
        self._error: DeepSeekError | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> WebModel | None:
        async with self._lock:
            now = self._clock()
            if now < self._expires_at:
                return self._model
            if self._error is not None and now < self._retry_at:
                raise DeepSeekError(self._error.biz_code, self._error.biz_msg)
            try:
                raw = await asyncio.wait_for(self._client.get_model_configs(), timeout=self.timeout)
                model = parse_default_model(raw)
            except (DeepSeekError, asyncio.TimeoutError) as exc:
                error = exc if isinstance(exc, DeepSeekError) else DeepSeekError(504, "Timed out loading DeepSeek model configuration")
                self._error = error
                self._retry_at = self._clock() + min(30.0, self.ttl)
                if error is exc:
                    raise
                raise error from exc
            self._model = model
            self._error = None
            self._expires_at = self._clock() + self.ttl
            return model


def advertised_models(models: Sequence[WebModel]) -> list[dict]:
    result = []
    for alias, thinking in MODEL_ALIASES.items():
        eligible = [model for model in models if not thinking or model.thinking]
        if not eligible:
            continue
        vision = any(model.supports_files and model.files.vision for model in eligible)
        result.append({
            "id": alias, "object": "model", "created": 0, "owned_by": "deepseek",
            "name": "DeepSeek Web" + (" Thinking" if thinking else ""), "model_type": MODEL_TYPE,
            "thinking_enabled": thinking, "input_modalities": ["text", "image"] if vision else ["text"],
            "capabilities": {
                "vision": vision, "files": any(model.supports_files for model in eligible),
                "thinking": any(model.thinking for model in eligible), "search": any(model.search for model in eligible),
                "files_with_search": any(model.supports_files and model.search and not model.files.conflict_with_search for model in eligible),
            },
        })
    return result
