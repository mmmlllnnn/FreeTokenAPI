"""DeepSeek parser recovery: actual upstream metadata, bounded retries, no data loss."""
from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from test_deepseek_models import fake_account, install_pool

from freetokenapi.api import openai as api
from freetokenapi.deepseek.client import DeepSeekClient, DeepSeekFileParseError

BUSY = {"id": "failed-file", "status": "FAILED", "error_code": 50404, "retryable": True, "audit_result": "unknown"}
READY = {"id": "ready-file", "status": "SUCCESS", "audit_result": "pass"}


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(api, "FILE_PARSE_RETRY_DELAY", 0)


def parse_client(*uploads):
    client = object.__new__(DeepSeekClient)
    client.upload_file = AsyncMock(side_effect=list(uploads))
    client.fetch_files = AsyncMock()
    return client


def account(client):
    return SimpleNamespace(client=client, index=0, mark_broken=Mock())


def attachment():
    return api.Attachment(b"%PDF-1.4 generated fixture", "generated.pdf", "application/pdf", False)


def test_busy_error_keeps_safe_metadata_without_misclassifying_it_as_bad_input():
    error = DeepSeekFileParseError({**BUSY, "signed_path": "PRIVATE_SIGNED_URL", "file_name": "PRIVATE_NAME"})
    assert error.biz_code == 503 and error.can_retry
    assert error.error_code == 50404 and error.upstream_retryable is True
    assert "busy" in str(error) and "FAILED" in str(error)
    assert "50404" in str(error) and "retryable=true" in str(error)
    assert "PRIVATE" not in str(error)


@pytest.mark.parametrize("thinking", [False, True])
async def test_retry_keeps_original_bytes_and_flags_but_uses_a_fresh_proof(monkeypatch, thinking):
    client = parse_client(BUSY, READY)
    proofs = AsyncMock(side_effect=[{"proof": "first"}, {"proof": "second"}])
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", proofs)
    file = attachment()
    assert await api._upload_attachments(account(client), [file], "default", thinking) == ["ready-file"]
    assert client.upload_file.await_count == 2
    for call in client.upload_file.await_args_list:
        assert call.args == (file.data, file.name, file.content_type, "default")
        assert call.kwargs["thinking_enabled"] is thinking
    assert client.upload_file.await_args_list[0].kwargs["pow_headers"] != client.upload_file.await_args_list[1].kwargs["pow_headers"]
    assert proofs.await_count == 2


async def test_repeated_busy_failure_stops_after_one_retry(monkeypatch):
    client = parse_client(BUSY, BUSY)
    acct = account(client)
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    with pytest.raises(HTTPException) as caught:
        await api._upload_attachments(acct, [attachment()], "default", False)
    assert caught.value.status_code == 503
    assert "50404" in caught.value.detail and "after one automatic retry" in caught.value.detail
    assert client.upload_file.await_count == 2
    acct.mark_broken.assert_not_called()


@pytest.mark.parametrize("info,expected", [
    ({"id": "f", "status": "FAILED"}, 502),
    ({**BUSY, "retryable": False}, 503),
    ({**BUSY, "retryable": None}, 503),
    ({**BUSY, "retryable": "true"}, 503),
    ({**BUSY, "error_code": 40001}, 400),
    ({**BUSY, "error_code": 40003}, 400),
    ({**BUSY, "status": "CONTENT_FILTER"}, 400),
    ({**BUSY, "status": "CONTENT_TOO_LONG"}, 400),
    ({**BUSY, "status": "CONTENT_EMPTY"}, 400),
    ({**BUSY, "status": "CANCELLED"}, 400),
    ({**BUSY, "audit_result": "reject"}, 400),
    ({**READY, "audit_result": "reject"}, 400),
    ({"id": "f", "audit_result": "reject"}, 400),
])
async def test_permanent_unapproved_or_filtered_failures_are_not_reuploaded(monkeypatch, info, expected):
    client = parse_client(info)
    acct = account(client)
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    with pytest.raises(HTTPException) as caught:
        await api._upload_attachments(acct, [attachment()], "default", False)
    assert caught.value.status_code == expected
    client.upload_file.assert_awaited_once()
    acct.mark_broken.assert_not_called()  # Parser 40001 is NOT an auth error.


async def test_poll_failure_preserves_retry_information():
    client = parse_client()
    client.fetch_files.return_value = [BUSY]
    with pytest.raises(DeepSeekFileParseError) as caught:
        await client.wait_for_file({"id": "failed-file", "status": "PARSING"})
    assert caught.value.error_code == 50404 and caught.value.can_retry


async def test_retry_delay_does_not_reset_total_budget(monkeypatch):
    client = parse_client(BUSY, READY)
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(api, "FILE_PARSE_RETRY_DELAY", 10)
    monkeypatch.setattr(api.settings, "file_parse_timeout", 0.01)
    with pytest.raises(HTTPException) as caught:
        await api._upload_attachments(account(client), [attachment()], "default", False)
    assert caught.value.status_code == 504
    client.upload_file.assert_awaited_once()


async def test_upload_is_also_bounded_by_preparation_deadline(monkeypatch):
    cancelled = asyncio.Event()
    async def slow(*_args, **_kwargs):
        try:
            await asyncio.sleep(20)
        finally:
            cancelled.set()
    client = parse_client()
    client.upload_file.side_effect = slow
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(api.settings, "file_parse_timeout", 0.01)
    with pytest.raises(HTTPException) as caught:
        await api._upload_attachments(account(client), [attachment()], "default", False)
    assert caught.value.status_code == 504 and cancelled.is_set()
    client.upload_file.assert_awaited_once()


async def test_parse_budget_is_not_the_http_request_timeout(monkeypatch):
    client = SimpleNamespace(upload_file=AsyncMock(return_value={"id": "f", "status": "PENDING"}), wait_for_file=AsyncMock(return_value=READY))
    monkeypatch.setattr(api, "_fresh_pow_upload_headers", AsyncMock(return_value={}))
    monkeypatch.setattr(api.settings, "timeout", 60)
    monkeypatch.setattr(api.settings, "file_parse_timeout", 180)
    assert await api._upload_attachments(account(client), [attachment()], "default", False) == ["ready-file"]
    client.wait_for_file.assert_awaited_once_with({"id": "f", "status": "PENDING"}, timeout=180)


@pytest.mark.parametrize("protocol", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recover", [False, True])
def test_three_protocols_do_not_generate_with_failed_file_references(monkeypatch, protocol, stream, recover):
    from deepseek_fixtures import model_configs

    acct = fake_account(0, model_configs())
    acct.client.upload_file.side_effect = [BUSY, READY if recover else BUSY]
    acct.client.wait_for_file = DeepSeekClient.wait_for_file.__get__(acct.client)
    _, http = install_pool(monkeypatch, [acct])
    data = base64.b64encode(attachment().data).decode()
    body = {"model": "deepseek-web", "stream": stream}
    if protocol == "responses":
        body["input"] = [{"role": "user", "content": [{"type": "input_file", "filename": "generated.pdf", "file_data": "data:application/pdf;base64," + data}]}]
    elif protocol == "messages":
        body.update(max_tokens=256, messages=[{"role": "user", "content": [{"type": "document", "title": "generated.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": data}}]}])
    else:
        body["messages"] = [{"role": "user", "content": [{"type": "file", "file": {"filename": "generated.pdf", "file_data": "data:application/pdf;base64," + data}}]}]
    response = http.post("/v1/" + protocol, json=body)
    assert acct.client.upload_file.await_count == 2
    if recover:
        assert response.status_code == 200 and "OK" in response.text, response.text
        assert acct.client.completion.await_args.kwargs["ref_file_ids"] == ["ready-file"]
    else:
        assert response.status_code == 503, response.text
        assert "50404" in response.text and "retryable=true" in response.text
        acct.client.completion.assert_not_awaited()


def test_unknown_retryable_failure_is_not_given_a_guessed_busy_cause():
    error = DeepSeekFileParseError({**BUSY, "error_code": 59999})
    assert error.can_retry and error.biz_code == 503
    assert "retryable file-parsing failure" in error.biz_msg
    assert "busy" not in error.biz_msg


def test_unexpected_metadata_does_not_leak_through_error_messages():
    error = DeepSeekFileParseError({"status": {"secret": "PRIVATE_URL"}, "error_code": "PRIVATE_URL", "audit_result": "reject"})
    assert error.biz_code == 400 and not error.can_retry
    assert "PRIVATE" not in str(error)
