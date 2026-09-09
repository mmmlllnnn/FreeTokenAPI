from __future__ import annotations

import asyncio
import base64
import copy
import datetime
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from freetokenapi import tools
from freetokenapi.accounts import AccountPool
from freetokenapi.api import openai as api
from freetokenapi.qwen import uploads
from freetokenapi.qwen.accounts import QwenAccount
from freetokenapi.qwen.client import QwenClient, QwenError

GRANT = {
    "access_key_id": "test-access-id", "access_key_secret": "test-access-secret", "security_token": "test-sts-token",
    "bucketname": "qwen-test-bucket", "region": "oss-us-east-1", "endpoint": "https://oss-us-east-1.aliyuncs.com",
    "file_id": "file-test", "file_path": "uploads/test file.txt", "file_url": "https://cdn.qwen.ai/test-file",
}


@pytest.fixture
async def upload_client(monkeypatch):
    web, storage = [], []
    state = {"grant": copy.deepcopy(GRANT), "parse_status": "success", "storage_status": 200}

    def web_handler(request):
        body = json.loads(request.content)
        web.append((request.url.path, body))
        assert request.headers["authorization"] == "Bearer fake-web-login"
        if request.url.path.endswith("getstsToken"):
            return httpx.Response(200, json={"success": True, "data": state["grant"]})
        if request.url.path.endswith("/parse"):
            return httpx.Response(200, json={"success": True, "data": {"file_id": "file-test"}})
        if request.url.path.endswith("/parse/status"):
            return httpx.Response(200, json={"success": True, "data": [{"file_id": "file-test", "status": state["parse_status"]}]})
        raise AssertionError(request.url.path)

    def storage_handler(request):
        storage.append(request)
        assert request.headers["authorization"].startswith("OSS4-HMAC-SHA256 Credential=test-access-id/")
        assert "fake-web-login" not in str(request.headers)
        assert "cookie" not in request.headers
        return httpx.Response(state["storage_status"], text="PRIVATE_OSS_ERROR_BODY")

    client = QwenClient("fake-web-login")
    await client.http.aclose()
    real_client = httpx.AsyncClient
    client.http = real_client(base_url="https://chat.qwen.ai", transport=httpx.MockTransport(web_handler), headers={"Authorization": "Bearer fake-web-login"})
    client.user_id = "test-user"

    def storage_client(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert "headers" not in kwargs and "cookies" not in kwargs
        return real_client(transport=httpx.MockTransport(storage_handler), **kwargs)

    monkeypatch.setattr(uploads.httpx, "AsyncClient", storage_client)
    monkeypatch.setattr(uploads, "PARSE_POLL_INTERVAL", 0)
    yield client, web, storage, state
    await client.aclose()


async def test_image_upload_uses_sts_oss_without_document_parse(upload_client):
    client, web, storage, _ = upload_client
    result = await client.upload_file(b"image-bytes", "photo.png", "image/png")
    assert web == [("/api/v2/files/getstsToken", {"filename": "photo.png", "filesize": "11", "filetype": "image"})]
    assert storage[0].content == b"image-bytes"
    assert result["type"] == "image" and result["file_class"] == "vision"
    assert result["file"]["user_id"] == "test-user"
    assert "access_key" not in json.dumps(result)


async def test_document_upload_waits_for_parse_and_reuses_cached_reference(upload_client):
    client, web, storage, _ = upload_client
    first = await client.upload_file(b"document", "note.txt", "text/plain")
    first["file"]["meta"]["name"] = "mutated copy"
    second = await client.upload_file(b"document", "note.txt", "text/plain")
    assert second["file"]["meta"]["name"] == "note.txt"
    assert second["file"]["meta"]["parse_meta"]["parse_status"] == "success"
    assert [path for path, _ in web] == ["/api/v2/files/getstsToken", "/api/v2/files/parse", "/api/v2/files/parse/status"]
    assert len(storage) == 1
    await client.upload_file(b"different", "note.txt", "text/plain")
    assert len(storage) == 2


async def test_failed_parse_is_not_cached(upload_client):
    client, web, _, state = upload_client
    state["parse_status"] = "failed"
    with pytest.raises(QwenError, match="could not parse"):
        await client.upload_file(b"bad document", "note.txt", "text/plain")
    state["parse_status"] = "success"
    await client.upload_file(b"bad document", "note.txt", "text/plain")
    assert sum(path.endswith("getstsToken") for path, _ in web) == 2


async def test_parse_timeout_is_explicit(upload_client, monkeypatch):
    client, _, _, _ = upload_client
    monkeypatch.setattr(uploads, "PARSE_TIMEOUT", 0)
    with pytest.raises(QwenError, match="timed out") as error:
        await client.upload_file(b"document", "note.txt", "text/plain")
    assert error.value.code == 504


async def test_oss_failure_does_not_leak_response_or_cache_file(upload_client):
    client, _, _, state = upload_client
    state["storage_status"] = 403
    with pytest.raises(QwenError) as error:
        await client.upload_file(b"data", "photo.png", "image/png")
    assert "PRIVATE_OSS_ERROR_BODY" not in str(error.value)
    assert not client._uploads


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1", "https://localhost", "https://oss-us-east-1.aliyuncs.com.evil.test",
    "https://user:password@oss-us-east-1.aliyuncs.com", "https://oss-us-east-1.aliyuncs.com:8443",
    "https://oss-us-east-1.aliyuncs.com/wrong-path", "https://oss-us-east-1.aliyuncs.com?query=secret",
])
async def test_untrusted_upload_endpoint_never_receives_file(upload_client, endpoint):
    client, _, storage, state = upload_client
    state["grant"]["endpoint"] = endpoint
    with pytest.raises(QwenError, match="upload endpoint"):
        await client.upload_file(b"data", "photo.png", "image/png")
    assert not storage


def test_oss_v4_signing_is_stable_and_encodes_object_path():
    now = datetime.datetime(2026, 9, 8, 1, 2, 3, tzinfo=datetime.timezone.utc)
    url, headers = uploads.oss_upload_request(GRANT, b"test", "text/plain", now=now)
    assert url == "https://qwen-test-bucket.oss-us-east-1.aliyuncs.com/uploads/test%20file.txt"
    assert headers["content-md5"] == "CY9rzUYh03PK3k6DJie09g=="
    assert headers["x-oss-date"] == "20260908T010203Z"
    assert headers["x-oss-security-token"] == "test-sts-token"
    assert "test-access-secret" not in str(headers)
    assert headers == uploads.oss_upload_request(GRANT, b"test", "text/plain", now=now)[1]
    assert headers["authorization"] != uploads.oss_upload_request(GRANT, b"changed", "text/plain", now=now)[1]["authorization"]


SSE = (
    'data: {"response.created":{"chat_id":"web-test","parent_id":null,"response_id":"r1"}}\n\n'
    'data: {"choices":[{"delta":{"role":"assistant","content":"Hello attachment","phase":"answer","status":"typing"}}],"response_id":"r1"}\n\n'
    'data: {"choices":[{"delta":{"role":"assistant","content":"","phase":"answer","status":"finished"}}],"response_id":"r1"}\n\n'
)


@pytest.fixture
def qwen_route(monkeypatch):
    sent = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"success": True, "data": {"id": "web-test"}})
        assert request.url.path == "/api/v2/chat/completions"
        sent.append(body)
        return httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"})

    client = QwenClient("fake-login")
    asyncio.run(client.http.aclose())
    client.http = httpx.AsyncClient(base_url="https://chat.qwen.ai", transport=httpx.MockTransport(handler))
    client.upload_file = AsyncMock(return_value={"id": "uploaded-file", "type": "image", "url": "https://cdn.qwen.ai/test"})
    account = QwenAccount(0, client)
    monkeypatch.setattr(api.app.state, "qwen_pool", AccountPool([account], label="qwen"), raising=False)
    monkeypatch.setattr(api.settings, "human_delay_min", 0)
    monkeypatch.setattr(api.settings, "human_delay_max", 0)
    yield TestClient(api.app), client.upload_file, sent
    asyncio.run(client.aclose())


def attachment_request(protocol, kind, stream):
    encoded = base64.b64encode(b"test attachment bytes").decode()
    mime = "image/png" if kind == "image" else "application/pdf"
    uri = f"data:{mime};base64,{encoded}"
    body = {"model": "qwen3.8-max", "stream": stream, "search": False}
    if protocol == "responses":
        part = {"type": "input_image", "image_url": uri} if kind == "image" else {"type": "input_file", "filename": "note.pdf", "file_data": uri}
        body["input"] = [{"role": "user", "content": [part]}]
    elif protocol == "messages":
        part = {"type": "image" if kind == "image" else "document", "source": {"type": "base64", "media_type": mime, "data": encoded}}
        body.update(messages=[{"role": "user", "content": [part]}], max_tokens=256)
    else:
        part = {"type": "image_url", "image_url": {"url": uri}} if kind == "image" else {"type": "file", "file": {"filename": "note.pdf", "file_data": uri}}
        body["messages"] = [{"role": "developer", "content": "Pi instructions"}, {"role": "user", "content": [part]}]
    return body


@pytest.mark.parametrize("protocol", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("kind", ["image", "file"])
@pytest.mark.parametrize("stream", [False, True])
def test_qwen_attachment_reaches_real_provider_pipeline(qwen_route, protocol, kind, stream):
    client, upload, sent = qwen_route
    response = client.post("/v1/" + protocol, json=attachment_request(protocol, kind, stream))
    assert response.status_code == 200, response.text
    assert "Hello attachment" in response.text
    upload.assert_awaited_once()
    assert upload.call_args.args[0] == b"test attachment bytes"
    assert upload.call_args.args[2] == ("image/png" if kind == "image" else "application/pdf")
    message = sent[0]["messages"][0]
    assert message["files"][0]["id"] == "uploaded-file"
    assert "base64," not in message["content"]
    assert "[Attached " in message["content"]


@pytest.mark.parametrize("protocol", ["chat/completions", "responses", "messages"])
def test_invalid_base64_does_not_start_upload(qwen_route, protocol):
    client, upload, sent = qwen_route
    body = attachment_request(protocol, "image", False)
    body = json.loads(json.dumps(body).replace(base64.b64encode(b"test attachment bytes").decode(), "not!base64"))
    response = client.post("/v1/" + protocol, json=body)
    assert response.status_code == 400
    upload.assert_not_awaited()
    assert not sent


def test_file_content_part_affects_context_fingerprint_and_not_prompt_bytes():
    def message(data):
        return api.ChatMessage(role="user", content=[{"type": "file", "file": {"filename": "note.txt", "file_data": data}}])
    a, b = message("YQ=="), message("Yg==")
    assert tools.context_sequence([a]) != tools.context_sequence([b])
    prompt, _ = tools.build_prompt([a])
    assert "note.txt" in prompt and "YQ==" not in prompt
