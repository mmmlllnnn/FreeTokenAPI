from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from deepseek_fixtures import model_configs
from fastapi.testclient import TestClient

from freetokenapi.accounts import AccountPool, DeepSeekAccount
from freetokenapi.api import openai as api
from freetokenapi.deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekSession
from freetokenapi.deepseek.models import (
    MODEL_ALIASES,
    ModelConfigCache,
    advertised_models,
    parse_default_model,
)


def test_unified_capabilities_and_aliases():
    model = parse_default_model(model_configs())
    assert model.model_type == "default"
    assert model.thinking and model.search and model.files.vision
    listed = advertised_models([model])
    assert [item["id"] for item in listed] == list(MODEL_ALIASES)
    assert all(item["input_modalities"] == ["text", "image"] for item in listed)
    assert all(item["capabilities"]["files_with_search"] for item in listed)


@pytest.mark.parametrize("entries", [[], [{"model_type": "expert", "enabled": True, "switchable": True}], model_configs(enabled=False), model_configs(switchable=False)])
def test_disabled_or_missing_default_is_not_advertised(entries):
    assert parse_default_model(entries) is None


@pytest.mark.parametrize("entries", [None, {}, [None], model_configs(enabled="false"), model_configs() * 2, model_configs(file_feature=True), model_configs(think_feature="enabled"), model_configs(file_feature={"vision": "false"}), model_configs(file_feature={"max_input_file_count": True}), model_configs(file_feature={"max_upload_file_size": -1}), model_configs(file_feature={"support_file_exts": "pdf"})])
def test_malformed_capabilities_are_not_guessed(entries):
    with pytest.raises(DeepSeekError):
        parse_default_model(entries)


def test_feature_presence_and_thinking_alias_filtering():
    plain = parse_default_model(model_configs(think_feature=None, search_feature=None, file_feature=None))
    assert not plain.thinking and not plain.search and plain.files is None
    data = advertised_models([plain])
    assert [item["id"] for item in data] == ["deepseek-web"]
    assert data[0]["input_modalities"] == ["text"]
    vision = parse_default_model(model_configs())
    assert advertised_models([plain, vision])[1]["capabilities"]["vision"]


async def test_cache_single_flight_and_expiry_observe_disabled_model():
    now = [100.0]
    client = SimpleNamespace(get_model_configs=AsyncMock(side_effect=[model_configs(), model_configs(enabled=False)]))
    cache = ModelConfigCache(client, ttl=10, clock=lambda: now[0])
    values = await asyncio.gather(*(cache.get() for _ in range(12)))
    assert all(value is values[0] for value in values)
    assert client.get_model_configs.await_count == 1
    now[0] += 11
    assert await cache.get() is None
    assert await cache.get() is None
    assert client.get_model_configs.await_count == 2


async def test_expired_cache_fails_closed_and_retries_after_cooldown():
    now = [100.0]
    client = SimpleNamespace(get_model_configs=AsyncMock(side_effect=[model_configs(), DeepSeekError(502, "offline"), model_configs(file_feature=None)]))
    cache = ModelConfigCache(client, ttl=5, clock=lambda: now[0])
    assert (await cache.get()).files is not None
    now[0] += 6
    for _ in range(2):
        with pytest.raises(DeepSeekError):
            await cache.get()
    assert client.get_model_configs.await_count == 2
    now[0] += 6
    assert (await cache.get()).files is None


async def test_cache_is_account_local():
    first = DeepSeekAccount(0, SimpleNamespace(get_model_configs=AsyncMock(return_value=model_configs())))
    second = DeepSeekAccount(1, SimpleNamespace(get_model_configs=AsyncMock(return_value=model_configs(think_feature=None))))
    assert (await first.model_config.get()).thinking
    assert not (await second.model_config.get()).thinking


async def test_fetch_timeout_does_not_leave_cache_locked():
    async def slow():
        await asyncio.sleep(30)
    client = SimpleNamespace(get_model_configs=AsyncMock(side_effect=slow))
    cache = ModelConfigCache(client, ttl=0, timeout=0.001)
    with pytest.raises(DeepSeekError) as caught:
        await cache.get()
    assert caught.value.biz_code == 504
    client.get_model_configs = AsyncMock(return_value=model_configs())
    assert (await cache.get()).model_type == "default"


@pytest.mark.parametrize("raw,code", [
    ({"code": 0, "data": {"biz_code": 0, "biz_data": {"settings": {"model_configs": {"id": 9, "value": model_configs()}}}}}, None),
    ({"code": 0, "data": {"biz_code": 40001, "biz_msg": "expired"}}, 40001),
    ({"code": 0, "data": []}, 502),
    ({"code": 0, "data": {"biz_data": {"settings": {}}}}, 502),
    ([], 502),
])
async def test_client_fetches_account_scoped_model_setting(monkeypatch, raw, code):
    client = DeepSeekClient("test-token", device_id="test-device")
    response = httpx.Response(200, json=raw, request=httpx.Request("GET", "https://chat.deepseek.com/api/v0/client/settings"))
    get = AsyncMock(return_value=response)
    monkeypatch.setattr(client.http, "get", get)
    try:
        if code is None:
            assert await client.get_model_configs() == model_configs()
        else:
            with pytest.raises(DeepSeekError) as caught:
                await client.get_model_configs()
            assert caught.value.biz_code == code
        get.assert_awaited_once_with("/api/v0/client/settings", params={"did": "test-device", "scope": "model"})
        assert client.http.headers["Authorization"] == "Bearer test-token"
    finally:
        await client.aclose()


def fake_account(index, configs):
    response = "data: " + json.dumps({"v": {"response": {"message_id": "reply", "status": "FINISHED", "fragments": [{"type": "RESPONSE", "content": "OK"}]}}}) + "\n\n"
    client = SimpleNamespace(
        get_model_configs=AsyncMock(return_value=configs),
        completion=AsyncMock(side_effect=lambda **_kwargs: httpx.Response(200, content=response, headers={"content-type": "text/event-stream"})),
        create_session=AsyncMock(return_value=DeepSeekSession(id=f"session-{index}")), create_pow_challenge=AsyncMock(),
        upload_file=AsyncMock(return_value={"id": f"file-{index}"}), stop_stream=AsyncMock(),
    )
    account = DeepSeekAccount(index, client)
    account.pow.make_header = AsyncMock(return_value={})
    account.pow_upload.make_header = AsyncMock(return_value={})
    return account


def install_pool(monkeypatch, accounts):
    pool = AccountPool(accounts)
    monkeypatch.setattr(api.app.state, "pool", pool, raising=False)
    monkeypatch.setattr(api.app.state, "qwen_pool", None, raising=False)
    monkeypatch.setattr(api.app.state, "qwen_models", [], raising=False)
    monkeypatch.setattr(api.app.state, "usage", None, raising=False)
    monkeypatch.setattr(api.settings, "search_enabled", False)
    monkeypatch.setattr(api, "_human_delay", AsyncMock())
    return pool, TestClient(api.app)


def test_models_api_uses_cached_account_capabilities(monkeypatch):
    first = fake_account(0, model_configs(think_feature=None, file_feature=None))
    second = fake_account(1, model_configs())
    _, http = install_pool(monkeypatch, [first, second])
    data = http.get("/v1/models").json()["data"]
    assert [entry["id"] for entry in data] == list(MODEL_ALIASES)
    assert data[0]["capabilities"]["vision"]
    assert http.get("/v1/models").status_code == 200
    first.client.get_model_configs.assert_awaited_once()
    second.client.get_model_configs.assert_awaited_once()


def test_models_api_does_not_invent_models_without_configuration(monkeypatch):
    _, http = install_pool(monkeypatch, [fake_account(0, model_configs(enabled=False))])
    assert http.get("/v1/models").json()["data"] == []


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-flash-thinking", "deepseek-v4-pro", "deepseek-v4-pro-thinking", "deepseek-v4-vision", "deepseek-v4-vision-thinking", "deepseek-web-extra"])
@pytest.mark.parametrize("protocol", ["chat/completions", "responses", "messages"])
def test_legacy_models_are_rejected_before_upstream_io(monkeypatch, model, protocol):
    account = fake_account(0, model_configs())
    _, http = install_pool(monkeypatch, [account])
    body = {"model": model, "input": "Hi"} if protocol == "responses" else {"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 100}
    assert http.post("/v1/" + protocol, json=body).status_code == 404
    account.client.get_model_configs.assert_not_awaited()
    account.client.completion.assert_not_awaited()


@pytest.mark.parametrize("protocol", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_document_request_uses_a_capable_account_and_default_mode(monkeypatch, protocol, stream):
    first = fake_account(0, model_configs(file_feature=None))
    second = fake_account(1, model_configs())
    _, http = install_pool(monkeypatch, [first, second])
    encoded = base64.b64encode(b"%PDF-1.4 fixture").decode()
    body = {"model": "deepseek-web-thinking", "stream": stream}
    if protocol == "responses":
        body["input"] = [{"role": "user", "content": [{"type": "input_text", "text": "Read it"}, {"type": "input_file", "filename": "report.pdf", "file_data": "data:application/pdf;base64," + encoded}]}]
    elif protocol == "messages":
        body.update(max_tokens=100, messages=[{"role": "user", "content": [{"type": "text", "text": "Read it"}, {"type": "document", "title": "report.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": encoded}}]}])
    else:
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "Read it"}, {"type": "file", "file": {"filename": "report.pdf", "file_data": "data:application/pdf;base64," + encoded}}]}]
    response = http.post("/v1/" + protocol, json=body)
    assert response.status_code == 200, response.text
    assert "OK" in response.text
    if protocol == "responses":
        assert '"type":"response.failed"' not in response.text
        if not stream:
            assert response.json()["error"] is None
    else:
        assert '"error"' not in response.text
    first.client.completion.assert_not_awaited()
    first.client.upload_file.assert_not_awaited()
    assert second.client.completion.await_args.kwargs["model_type"] == "default"
    assert second.client.completion.await_args.kwargs["thinking_enabled"]
    assert second.client.upload_file.await_args.args[3] == "default"


@pytest.mark.parametrize("changes,options", [
    ({"think_feature": None}, {"thinking": True}),
    ({"search_feature": None}, {"search": True}),
])
def test_unsupported_requested_features_are_not_silently_disabled(monkeypatch, changes, options):
    account = fake_account(0, model_configs(**changes))
    _, http = install_pool(monkeypatch, [account])
    response = http.post("/v1/chat/completions", json={"model": "deepseek-web", "messages": [{"role": "user", "content": "Hi"}], **options})
    assert response.status_code == 400
    account.client.completion.assert_not_awaited()


def test_file_constraints_follow_model_configuration():
    cfg = model_configs()
    cfg[0]["file_feature"].update(max_input_file_count=1, max_upload_file_size=10, support_file_exts=["PDF"], conflict_with_search=True)
    model = parse_default_model(cfg)
    good = api.Attachment(b"123", "a.pdf", "application/pdf", False)
    assert model.rejection(thinking=False, search=False, attachments=[good]) is None
    assert model.rejection(thinking=False, search=True, attachments=[good])
    assert model.rejection(thinking=False, search=False, attachments=[good, good])
    assert model.rejection(thinking=False, search=False, attachments=[api.Attachment(b"x" * 11, "a.pdf", "application/pdf", False)])
    assert model.rejection(thinking=False, search=False, attachments=[api.Attachment(b"x", "a.zip", "application/zip", False)])
    cfg[0]["file_feature"]["vision"] = False
    cfg[0]["file_feature"]["support_file_exts"] = ["png"]
    assert parse_default_model(cfg).rejection(thinking=False, search=False, attachments=[api.Attachment(b"x", "a.png", "image/png", True)])


async def test_account_affinity_cannot_override_capability_filter():
    first, second = fake_account(0, model_configs()), fake_account(1, model_configs())
    pool = AccountPool([first, second])
    pool.register(0, "old-session")
    selected, session_id = await pool.acquire("old-session", allowed_indices={1})
    assert selected is second and session_id is None
    await second.sem.acquire()
    try:
        from freetokenapi.accounts import AccountPoolBusy
        with pytest.raises(AccountPoolBusy):
            await pool.acquire(None, max_wait=0, allowed_indices={1})
    finally:
        second.sem.release()


def test_metadata_failure_does_not_remove_qwen_models(monkeypatch):
    account = fake_account(0, model_configs())
    account.client.get_model_configs.side_effect = DeepSeekError(502, "network error")
    _, http = install_pool(monkeypatch, [account])
    assert http.get("/v1/models").status_code == 502
    monkeypatch.setattr(api.app.state, "qwen_models", [{"id": "qwen3.8-max"}])
    assert [model["id"] for model in http.get("/v1/models").json()["data"]] == ["qwen3.8-max"]


@pytest.mark.parametrize("field", ["max_input_file_count", "max_upload_file_size"])
def test_zero_upload_quota_is_not_advertised_as_file_support(field):
    cfg = model_configs()
    cfg[0]["file_feature"][field] = 0
    model = parse_default_model(cfg)
    assert not model.supports_files
    entry = advertised_models([model])[0]
    assert not entry["capabilities"]["files"] and not entry["capabilities"]["vision"]
    assert entry["input_modalities"] == ["text"]


async def test_cancelled_refresh_does_not_poison_later_requests():
    started = asyncio.Event()
    async def blocked():
        started.set()
        await asyncio.Event().wait()
    client = SimpleNamespace(get_model_configs=AsyncMock(side_effect=blocked))
    cache = ModelConfigCache(client)
    pending = asyncio.create_task(cache.get())
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    client.get_model_configs = AsyncMock(return_value=model_configs())
    assert (await cache.get()).thinking


def test_explicit_client_session_key_still_reuses_the_same_web_session(monkeypatch):
    account = fake_account(0, model_configs())
    _, http = install_pool(monkeypatch, [account])
    body = {"model": "deepseek-web", "session_id": "client-key", "messages": [{"role": "user", "content": "First"}]}
    first = http.post("/v1/chat/completions", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["session_id"] == "client-key"
    body["messages"] = [{"role": "user", "content": "Next"}]
    second = http.post("/v1/chat/completions", json=body)
    assert second.status_code == 200, second.text
    assert second.json()["session_id"] == "client-key"
    account.client.create_session.assert_awaited_once()
    assert account.client.completion.await_args.kwargs["prompt"] == "Next"


def test_cross_account_capability_routing_builds_fresh_history(monkeypatch):
    first = fake_account(0, model_configs(file_feature=None))
    second = fake_account(1, model_configs())
    _, http = install_pool(monkeypatch, [first, second])
    body = {"model": "deepseek-web", "session_id": "client-key", "messages": [{"role": "user", "content": "Remember the original request"}]}
    assert http.post("/v1/chat/completions", json=body).status_code == 200
    body["messages"] += [{"role": "assistant", "content": "Remembered"}, {"role": "user", "content": [{"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,eA=="}}]}]
    response = http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["session_id"] == "session-1"
    assert second.client.completion.await_args.kwargs["parent_message_id"] is None
    assert "Remember the original request" in second.client.completion.await_args.kwargs["prompt"]
    first.client.upload_file.assert_not_awaited()
    second.client.create_session.assert_awaited_once()


def test_no_single_account_matches_the_combined_requirements(monkeypatch):
    first = fake_account(0, model_configs(file_feature=None))
    second = fake_account(1, model_configs(think_feature=None))
    _, http = install_pool(monkeypatch, [first, second])
    response = http.post("/v1/chat/completions", json={"model": "deepseek-web-thinking", "messages": [{"role": "user", "content": [{"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,eA=="}}]}]})
    assert response.status_code == 400
    first.client.completion.assert_not_awaited()
    second.client.completion.assert_not_awaited()
    first.client.upload_file.assert_not_awaited()
    second.client.upload_file.assert_not_awaited()
