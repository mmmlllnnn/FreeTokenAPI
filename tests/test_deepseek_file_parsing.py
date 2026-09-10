from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from deepseek_fixtures import model_configs
from fastapi import HTTPException

from freetokenapi.api import openai as api
from freetokenapi.deepseek.client import DeepSeekClient, DeepSeekError
from freetokenapi.deepseek.models import parse_default_model


def client_with_statuses(*records):
    client = object.__new__(DeepSeekClient)
    client.fetch_files = AsyncMock(side_effect=list(records))
    return client


@pytest.mark.parametrize("info", [{"id": "f1"}, {"id": "f1", "status": "SUCCESS"}])
async def test_ready_or_legacy_files_do_not_poll(info):
    client = client_with_statuses()
    assert await client.wait_for_file(info) == info
    client.fetch_files.assert_not_awaited()


async def test_pending_file_waits_for_matching_success():
    client = client_with_statuses([{"id": "other", "status": "FAILED"}, {"id": "f1", "status": "SUCCESS"}])
    info = await client.wait_for_file({"id": "f1", "status": "PENDING"})
    assert info["status"] == "SUCCESS"
    client.fetch_files.assert_awaited_once_with(["f1"])


@pytest.mark.parametrize("status,code", [("FAILED", 502), ("CONTENT_FILTER", 400), ("CONTENT_TOO_LONG", 400), ("CANCELLED", 400), ("CONTENT_EMPTY", 400)])
async def test_parse_failure_is_not_sent_to_generation(status, code):
    client = client_with_statuses([{"id": "f1", "status": status}])
    with pytest.raises(DeepSeekError) as error:
        await client.wait_for_file({"id": "f1", "status": "PARSING"})
    assert error.value.biz_code == code
    assert status in error.value.biz_msg


async def test_parse_timeout_is_bounded():
    client = client_with_statuses()
    with pytest.raises(DeepSeekError) as error:
        await client.wait_for_file({"id": "f1", "status": "PENDING"}, timeout=0)
    assert error.value.biz_code == 504
    client.fetch_files.assert_not_awaited()


async def test_network_poll_respects_parse_deadline():
    client = client_with_statuses()

    async def slow(_ids):
        await asyncio.sleep(10)

    client.fetch_files.side_effect = slow
    with pytest.raises(DeepSeekError) as error:
        await client.wait_for_file({"id": "f1", "status": "PARSING"}, timeout=0.01)
    assert error.value.biz_code == 504


@pytest.mark.parametrize("info", [{"status": "PENDING"}, {"id": "f1", "status": []}])
async def test_bad_parse_metadata_is_a_gateway_error(info):
    client = client_with_statuses()
    with pytest.raises(DeepSeekError) as error:
        await client.wait_for_file(info)
    assert error.value.biz_code == 502


async def test_api_upload_waits_before_returning_file_ids(monkeypatch):
    pending = {"id": "f1", "status": "PARSING"}
    client = SimpleNamespace(upload_file=AsyncMock(return_value=pending), wait_for_file=AsyncMock(return_value={"id": "f1", "status": "SUCCESS"}))
    account = SimpleNamespace(client=client)
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={"X-DS-PoW-Response": "fake"}))
    ids = await api._upload_attachments(account, [api.Attachment(b"test", "report.pdf", "application/pdf", False)], "default", True)
    assert ids == ["f1"]
    client.wait_for_file.assert_awaited_once_with(pending, timeout=api.settings.file_parse_timeout)


@pytest.mark.parametrize("code,status", [(400, 400), (503, 503), (504, 504), (502, 502), (40001, 401)])
async def test_api_returns_parse_errors_instead_of_generating(monkeypatch, code, status):
    client = SimpleNamespace(upload_file=AsyncMock(return_value={"id": "f1", "status": "PARSING"}), wait_for_file=AsyncMock(side_effect=DeepSeekError(code, "attachment parsing failed")))
    account = SimpleNamespace(client=client)
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(api, "_handle_account_error", lambda *_args: None)
    with pytest.raises(HTTPException) as error:
        await api._upload_attachments(account, [api.Attachment(b"test", "report.pdf", "application/pdf", False)], "default", True)
    assert error.value.status_code == status


@pytest.mark.parametrize("error_code,expected_calls", [(40301, 2), (400, 1), (429, 1)])
async def test_upload_retries_only_a_rejected_pow_once(monkeypatch, error_code, expected_calls):
    client = SimpleNamespace(upload_file=AsyncMock(side_effect=[DeepSeekError(error_code, "rejected"), {"id": "f1"}]))
    account = SimpleNamespace(client=client)
    fresh = AsyncMock(side_effect=[{"proof": "one"}, {"proof": "two"}])
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", fresh)
    monkeypatch.setattr(api, "_handle_account_error", lambda *_args: None)
    attachments = [api.Attachment(b"data", "report.pdf", "application/pdf", False)]
    if error_code == 40301:
        assert await api._upload_attachments(account, attachments, "default", False) == ["f1"]
        assert client.upload_file.await_args_list[0].kwargs["pow_headers"] != client.upload_file.await_args_list[1].kwargs["pow_headers"]
    else:
        with pytest.raises(HTTPException):
            await api._upload_attachments(account, attachments, "default", False)
    assert client.upload_file.await_count == fresh.await_count == expected_calls


async def test_repeated_invalid_pow_is_not_retried_indefinitely(monkeypatch):
    account = SimpleNamespace(client=SimpleNamespace(upload_file=AsyncMock(side_effect=DeepSeekError(40301, "invalid proof"))))
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(api, "_handle_account_error", lambda *_args: None)
    with pytest.raises(HTTPException):
        await api._upload_attachments(account, [api.Attachment(b"data", "report.pdf", "application/pdf", False)], "default", False)
    assert account.client.upload_file.await_count == 2


@pytest.mark.parametrize("stream", [False, True])
async def test_busy_upload_returns_429_before_generation(monkeypatch, stream):
    account = SimpleNamespace(sem=asyncio.Semaphore(0))
    upload = AsyncMock()
    monkeypatch.setattr(api.app.state, "pool", object(), raising=False)
    monkeypatch.setattr(api.settings, "acquire_timeout", 0.01)
    monkeypatch.setattr(api, "_acquire_and_build", AsyncMock(return_value=(account, None, (), "read the file", False)))
    monkeypatch.setattr(api, "_upload_attachments", upload)
    monkeypatch.setattr(api, "_deepseek_model_configs", AsyncMock(return_value=({0: parse_default_model(model_configs())}, [])))
    request = api.ChatCompletionRequest(
        model="deepseek-web", stream=stream,
        messages=[{"role": "user", "content": [{"type": "file", "file": {"filename": "report.txt", "file_data": "data:text/plain;base64,dGVzdA=="}}]}],
    )
    with pytest.raises(HTTPException) as error:
        await api._chat_completions_deepseek(request)
    assert error.value.status_code == 429
    upload.assert_not_awaited()
