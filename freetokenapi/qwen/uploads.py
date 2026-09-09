"""Qwen web uploads: v2 STS grant, OSS V4 PUT, then document parsing.

Wire format checked against the public Qwen web frontend 0.2.91. Storage
requests use a separate HTTP client: web login cookies/Authorization must never
be sent to OSS, and redirects must not forward temporary upload credentials.
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import hmac
import re
import time
from urllib.parse import quote, urlsplit

import httpx

from .client import QwenError

PARSE_POLL_INTERVAL = 0.5
PARSE_TIMEOUT = 60.0


def _required(grant: dict, key: str) -> str:
    value = grant.get(key)
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n"):
        raise QwenError(502, "Qwen returned an incomplete upload grant")
    return value


def oss_upload_request(grant: dict, data: bytes, content_type: str, *, now: datetime.datetime | None = None) -> tuple[str, dict[str, str]]:
    """Sign a single-object PUT using Alibaba OSS Signature V4 (no SDK needed)."""
    bucket = _required(grant, "bucketname")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
        raise QwenError(502, "Qwen returned an invalid upload bucket")
    endpoint = _required(grant, "endpoint")
    endpoint = endpoint if "://" in endpoint else "https://" + endpoint
    parsed = urlsplit(endpoint)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or not (host.endswith(".aliyuncs.com") or host.endswith(".aliyuncs.com.cn"))):
        raise QwenError(502, "Qwen returned an untrusted upload endpoint")
    if not host.startswith(bucket + "."):
        host = bucket + "." + host
    path = _required(grant, "file_path").lstrip("/")
    if not path or any(part in {".", ".."} for part in path.split("/")):
        raise QwenError(502, "Qwen returned an invalid upload object path")
    region = _required(grant, "region").removeprefix("oss-")
    if not re.fullmatch(r"[a-z0-9-]+", region):
        raise QwenError(502, "Qwen returned an invalid upload region")
    access_id = _required(grant, "access_key_id")
    secret = _required(grant, "access_key_secret")
    token = _required(grant, "security_token")
    if any(c in content_type for c in "\r\n"):
        raise QwenError(400, "Invalid attachment content type")
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    day = stamp[:8]
    headers = {
        "content-type": content_type,
        "content-md5": base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode(),
        "x-oss-content-sha256": "UNSIGNED-PAYLOAD",
        "x-oss-date": stamp,
        "x-oss-security-token": token,
    }
    canonical_headers = "".join(f"{key}:{headers[key].strip()}\n" for key in sorted(headers))
    canonical = "\n".join(("PUT", quote(f"/{bucket}/{path}", safe="/~"), "", canonical_headers, "", "UNSIGNED-PAYLOAD"))
    scope = f"{day}/{region}/oss/aliyun_v4_request"
    to_sign = "\n".join(("OSS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()))
    key = ("aliyun_v4" + secret).encode()
    for part in (day, region, "oss", "aliyun_v4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = f"OSS4-HMAC-SHA256 Credential={access_id}/{scope},Signature={signature}"
    return f"https://{host}/{quote(path, safe='/~')}", headers


async def put_object(grant: dict, data: bytes, content_type: str, timeout: float) -> None:
    try:
        url, headers = oss_upload_request(grant, data, content_type)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as storage:
            response = await storage.put(url, headers=headers, content=data)
        if response.status_code not in (200, 201, 204):
            # OSS error XML can contain credential/canonical-request details.
            raise QwenError(502, f"Qwen attachment storage upload failed (HTTP {response.status_code})")
    except httpx.HTTPError as exc:
        raise QwenError(502, "Qwen attachment storage upload failed") from exc
    except ValueError as exc:
        raise QwenError(502, "Qwen returned an invalid upload endpoint") from exc


async def wait_for_parse(client, file_id: str) -> None:
    started = await client._post("/api/v2/files/parse", {"file_id": file_id})
    if started.get("file_id") != file_id:
        raise QwenError(502, "Qwen did not acknowledge attachment parsing")
    deadline = time.monotonic() + PARSE_TIMEOUT
    while time.monotonic() < deadline:
        response = await client.http.post(
            "/api/v2/files/parse/status", json={"file_id_list": [file_id]}, headers=client._request_headers()
        )
        if response.status_code != 200:
            raise QwenError(response.status_code, "Qwen attachment parsing status request failed")
        try:
            payload = response.json()
        except ValueError as exc:
            raise QwenError(502, "Qwen returned an invalid attachment parsing status") from exc
        if not isinstance(payload, dict):
            raise QwenError(502, "Qwen returned an invalid attachment parsing status")
        if payload.get("success") is False:
            client._biz(payload)
        records = payload.get("data")
        if isinstance(records, dict):
            records = records.get("data")
        if not isinstance(records, list):
            raise QwenError(502, "Qwen returned an invalid attachment parsing status")
        for record in records:
            if not isinstance(record, dict) or record.get("file_id") != file_id:
                continue
            if record.get("status") == "success":
                return
            if record.get("status") == "failed":
                raise QwenError(400, "Qwen could not parse this attachment; check the file format and model support")
        await asyncio.sleep(min(PARSE_POLL_INTERVAL, max(0.0, deadline - time.monotonic())))
    raise QwenError(504, "Qwen attachment parsing timed out")


async def upload_file(client, data: bytes, filename: str, content_type: str) -> dict:
    kind = next((kind for kind in ("image", "audio", "video") if content_type.startswith(kind + "/")), "file")
    grant = await client._post("/api/v2/files/getstsToken", {"filename": filename, "filesize": str(len(data)), "filetype": kind})
    file_id = _required(grant, "file_id")
    url = _required(grant, "file_url")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise QwenError(502, "Qwen returned an invalid attachment URL") from exc
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise QwenError(502, "Qwen returned an invalid attachment URL")
    await put_object(grant, data, content_type, client.timeout)
    if kind == "file":
        await wait_for_parse(client, file_id)
    stamp = int(time.time() * 1000)
    meta = {"name": filename, "size": len(data), "content_type": content_type}
    if kind == "file":
        meta["parse_meta"] = {"parse_status": "success"}
    file = {"id": file_id, "filename": filename, "hash": None, "data": {}, "meta": meta, "created_at": stamp, "update_at": stamp}
    if client.user_id:
        file["user_id"] = client.user_id
    return {
        "type": kind, "id": file_id, "url": url, "name": filename, "file": file,
        "size": len(data), "file_type": content_type, "status": "uploaded", "progress": 100,
        "file_class": {"image": "vision", "file": "document"}.get(kind, kind),
        "showType": kind, "collection_name": "", "error": "",
    }
