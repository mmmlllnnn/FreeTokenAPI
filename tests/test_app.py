from pathlib import Path
from unittest.mock import Mock

import pytest
import uvicorn
from fastapi.testclient import TestClient

import app as launcher
from freetokenapi import __version__
from freetokenapi.api.openai import app as api
from freetokenapi.config import settings


@pytest.fixture
def project_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher.sys, "path", list(launcher.sys.path))
    monkeypatch.delenv("DEEPSEEK_TOKENS", raising=False)
    monkeypatch.delenv("QWEN_TOKENS", raising=False)
    return tmp_path


@pytest.fixture
def run_server(monkeypatch, project_dir):
    (project_dir / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(settings, "deepseek_tokens", [])
    monkeypatch.setattr(settings, "qwen_tokens", ["test-qwen-token"])
    monkeypatch.setattr(settings, "host", "127.0.0.1")
    monkeypatch.setattr(settings, "port", 8000)
    monkeypatch.setattr("freetokenapi.logging.uvicorn_log_config", lambda: {"version": 1})
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    return run


def test_first_run_copies_example_on_python_310_plus(project_dir, capsys):
    example = "# 登录令牌\nDEEPSEEK_TOKENS=\nQWEN_TOKENS=\n"
    (project_dir / ".env.example").write_text(example, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        launcher.ensure_env()
    assert exc.value.code == 1
    assert (project_dir / ".env").read_text(encoding="utf-8") == example
    assert "DEEPSEEK_TOKENS" in capsys.readouterr().out


def test_existing_env_is_not_overwritten(project_dir):
    env_file = project_dir / ".env"
    env_file.write_text("QWEN_TOKENS=keep-me\n", encoding="utf-8")
    (project_dir / ".env.example").write_text("QWEN_TOKENS=\n", encoding="utf-8")
    launcher.ensure_env()
    assert env_file.read_text(encoding="utf-8") == "QWEN_TOKENS=keep-me\n"


@pytest.mark.parametrize("key", ["DEEPSEEK_TOKENS", "QWEN_TOKENS"])
def test_environment_credentials_do_not_require_env_file(project_dir, monkeypatch, key):
    monkeypatch.setenv(key, "test-token")
    launcher.ensure_env()
    assert not (project_dir / ".env").exists()


def test_missing_env_and_template_fail_clearly(project_dir, capsys):
    with pytest.raises(SystemExit) as exc:
        launcher.ensure_env()
    assert exc.value.code == 1
    assert ".env.example" in capsys.readouterr().err


def test_empty_tokens_fail_before_server_start(monkeypatch, run_server, capsys):
    monkeypatch.setattr(settings, "qwen_tokens", [])
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 1
    run_server.assert_not_called()
    assert "DEEPSEEK_TOKENS / QWEN_TOKENS" in capsys.readouterr().err


def test_qwen_only_does_not_need_node(monkeypatch, run_server):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    launcher.main()
    run_server.assert_called_once_with(
        "freetokenapi.api.openai:app", host="127.0.0.1", port=8000, log_config={"version": 1}
    )


def test_deepseek_without_node_or_native_solver_fails(monkeypatch, run_server, capsys):
    monkeypatch.setattr(settings, "deepseek_tokens", ["test-deepseek-token"])
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr("freetokenapi.pow._find_native_solver", lambda: None)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 1
    run_server.assert_not_called()
    assert "Node.js" in capsys.readouterr().err


def test_deepseek_with_node_can_start(monkeypatch, run_server):
    monkeypatch.setattr(settings, "deepseek_tokens", ["test-deepseek-token"])
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "node")
    launcher.main()
    run_server.assert_called_once()


def test_native_solver_is_still_supported(monkeypatch, run_server):
    monkeypatch.setattr(settings, "deepseek_tokens", ["test-deepseek-token"])
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr("freetokenapi.pow._find_native_solver", lambda: Path("pow_solver.exe"))
    launcher.main()
    run_server.assert_called_once()


def test_non_loopback_bind_warns_about_missing_auth(monkeypatch, run_server, capsys):
    monkeypatch.setattr(settings, "host", "0.0.0.0")
    launcher.main()
    assert "鉴权" in capsys.readouterr().err
    assert run_server.call_args.kwargs["host"] == "0.0.0.0"


def test_old_python_fails_before_start(monkeypatch, run_server, capsys):
    monkeypatch.setattr(launcher.sys, "version_info", (3, 9, 0))
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 1
    run_server.assert_not_called()
    assert "Python 3.10+" in capsys.readouterr().err


@pytest.mark.parametrize("path", ["/", "/v1", "/v1/"])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_base_urls_return_lightweight_status_without_redirects(path, method):
    response = TestClient(api).request(method, path, follow_redirects=False)
    assert response.status_code == 200
    assert "location" not in response.headers
    if method == "GET":
        assert response.json() == {
            "service": "FreeTokenAPI",
            "version": "2.0.1",
            "status": "ok",
            "api": "/v1",
            "adapter_features": [
                "responses_namespace_tools", "messages_inline_instructions", "chat_completions_developer",
                "qwen_chat_attachments", "native_web_search", "agent_tool_continuation", "deepseek_model_configs", "deepseek_file_parse_recovery",
            ],
        }
    else:
        assert response.content == b""


@pytest.mark.parametrize("path", ["/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"])
def test_automatic_documentation_routes_are_disabled(path):
    assert TestClient(api).get(path).status_code == 404
    assert "docs" not in TestClient(api).get("/").json()


def test_openapi_keeps_chat_and_models_endpoints():
    schema = api.openapi()  # Internal schema inspection only; no public docs route.
    assert schema["info"]["title"] == "FreeTokenAPI"
    assert schema["info"]["version"] == __version__ == "2.0.1"
    paths = schema["paths"]
    assert "post" in paths["/v1/chat/completions"]
    assert "get" in paths["/v1/models"]


def test_release_runtime_resources_are_present():
    from freetokenapi.pow import _NODE_SOLVER, _SOLVER_DIR
    from freetokenapi.store import DEFAULT_CACHE_SUBDIR

    assert __version__ == "2.0.1"
    assert DEFAULT_CACHE_SUBDIR == "freetokenapi"
    assert _SOLVER_DIR == Path(__file__).resolve().parents[1] / "freetokenapi" / "deepseek"
    assert _NODE_SOLVER.is_file()
    assert (_SOLVER_DIR / "pow_solver.c").is_file()
    assert (_SOLVER_DIR / "sha3_wasm_bg.wasm").read_bytes()[:4] == bytes([0, 97, 115, 109])


def test_shipped_env_template_has_safe_defaults():
    from dotenv import dotenv_values

    template = Path(__file__).resolve().parents[1] / ".env.example"
    values = dotenv_values(template)
    assert values["DEEPSEEK_TOKENS"] == ""
    assert values["QWEN_TOKENS"] == ""
    assert values["FREETOKENAPI_HOST"] == "127.0.0.1"
    assert values["FREETOKENAPI_PORT"] == "8000"
    assert values["FREETOKENAPI_TIMEOUT"] == "60"
    assert all(key.startswith("FREETOKENAPI_") or key in {"DEEPSEEK_TOKENS", "QWEN_TOKENS"} for key in values)
