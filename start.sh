#!/bin/sh
# Portable macOS/Linux launcher. Never deletes or overwrites a user's environment.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$ROOT"
export PYTHONUTF8=1
PYTHON="$ROOT/.venv/bin/python"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

if [ ! -x "$PYTHON" ]; then
    if [ -e "$ROOT/.venv" ]; then
        fail "The existing .venv is not usable on this OS. Recreate it with Python 3.10+ before retrying; it has not been deleted."
    fi
    SYSTEM_PYTHON=
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
            SYSTEM_PYTHON=$candidate
            break
        fi
    done
    [ -n "$SYSTEM_PYTHON" ] || fail "Python 3.10+ was not found. Install Python and add it to PATH."
    printf '%s\n' 'Creating a local Python virtual environment...'
    "$SYSTEM_PYTHON" -m venv "$ROOT/.venv" ||
        fail "Could not create .venv. On some Linux distributions, install the matching python3-venv package first."
fi

"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1 ||
    fail "The project's Python environment is broken or older than Python 3.10. Recreate .venv before retrying."

if ! "$PYTHON" -c 'import fastapi, uvicorn, httpx, dotenv; from pydantic import field_validator; from PIL import Image' >/dev/null 2>&1; then
    printf '%s\n' 'Installing Python dependencies into .venv...'
    "$PYTHON" -m pip install -r "$ROOT/requirements.txt" || fail "Dependency installation failed. Check your network and Python/pip setup."
fi

# app.py initializes .env if absent and checks credentials and the DeepSeek PoW runtime.
# exec keeps Ctrl+C and the server exit status intact.
exec "$PYTHON" "$ROOT/app.py"
