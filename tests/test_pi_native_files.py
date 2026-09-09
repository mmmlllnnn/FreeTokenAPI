from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Pi native-file extension requires Node.js")
def test_pi_native_file_extension():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [shutil.which("node"), "--test", str(root / "tests" / "pi-native-files.test.mjs")],
        cwd=root, env={**os.environ, "NO_COLOR": "1"}, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
