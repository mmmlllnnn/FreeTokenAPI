from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
ROOT = Path(__file__).resolve().parent


def ensure_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        return
    if os.environ.get("DEEPSEEK_TOKENS", "").strip() or os.environ.get("QWEN_TOKENS", "").strip():
        return
    example = ROOT / ".env.example"
    if example.exists():
        shutil.copyfile(example, env_file)
        print("已从 .env.example 创建 .env。")
        print("请填入 DEEPSEEK_TOKENS 或 QWEN_TOKENS，然后重新启动。")
        sys.exit(1)
    print("找不到 .env 或 .env.example，请参照 README.md 创建配置。", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if sys.version_info < MIN_PYTHON:
        print(
            f"FreeTokenAPI requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, got {sys.version.split()[0]}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    ensure_env()

    try:
        import uvicorn
        from dotenv import load_dotenv
    except ImportError:
        print("缺少 Python 依赖。请运行 start.bat，或用当前 Python 执行 -m pip install -r requirements.txt。", file=sys.stderr)
        sys.exit(1)

    load_dotenv(ROOT / ".env", override=False)

    from freetokenapi import __version__
    from freetokenapi.config import settings
    from freetokenapi.logging import uvicorn_log_config

    if not settings.deepseek_tokens and not settings.qwen_tokens:
        print("尚未配置网页登录 Token：请编辑 .env，至少填写 DEEPSEEK_TOKENS / QWEN_TOKENS 其中一项。", file=sys.stderr)
        print("获取方法见 README.md；只填 Authorization 中 Bearer 后面的部分。", file=sys.stderr)
        sys.exit(1)

    if settings.deepseek_tokens and shutil.which("node") is None:
        from freetokenapi.pow import _find_native_solver

        if _find_native_solver() is None:
            print("使用 DeepSeek 需要 Node.js 计算 PoW。请安装 Node.js LTS，重新打开终端并确认 node --version 可用。", file=sys.stderr)
            sys.exit(1)

    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        print("警告：此服务没有 API Key 鉴权；监听非本机地址可能让其他设备使用你的账号。", file=sys.stderr)
    print(f"FreeTokenAPI {__version__} starting on {settings.host}:{settings.port} (正在校验网页登录 Token)")
    uvicorn.run(
        "freetokenapi.api.openai:app",
        host=settings.host,
        port=settings.port,
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
