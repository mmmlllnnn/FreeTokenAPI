from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def posix_shell():
    if os.name != "nt":
        return shutil.which("sh")
    git = shutil.which("git")
    candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe" if git else None
    return str(candidate) if candidate and candidate.is_file() else None


SHELL = posix_shell()
pytestmark = pytest.mark.skipif(not SHELL, reason="A POSIX shell is needed for launcher execution tests")

STUB = r"""#!/bin/sh
printf '%s|%s\n' "$PWD" "$*" >> "$TEST_TRACE"
if [ "$1" = '-c' ]; then
    case "$2" in
        *sys.version_info*) [ "${TEST_OLD:-0}" = 0 ]; exit $? ;;
        *) if [ "${TEST_MISSING_DEPS:-0}" = 1 ] && [ ! -f "$TEST_INSTALLED" ]; then exit 1; fi; exit 0 ;;
    esac
fi
if [ "$1" = '-m' ] && [ "$2" = 'venv' ]; then
    mkdir -p "$3/bin"
    cp "$TEST_TEMPLATE" "$3/bin/python"
    chmod +x "$3/bin/python"
    exit 0
fi
if [ "$1" = '-m' ] && [ "$2" = 'pip' ]; then
    [ "${TEST_PIP_FAIL:-0}" = 0 ] || exit 2
    : > "$TEST_INSTALLED"
    exit 0
fi
exit "${TEST_APP_EXIT:-0}"
"""


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    shutil.copyfile(ROOT / "start.sh", root / "start.sh")
    (root / "app.py").write_text("# Stubbed in launcher tests", encoding="utf-8")
    (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    template = tmp_path / "python-stub"
    template.write_text(STUB, encoding="utf-8", newline="\n")
    template.chmod(0o755)
    for name in ("python3", "python"):
        target = bin_dir / name
        shutil.copyfile(template, target)
        target.chmod(0o755)
    trace = tmp_path / "trace.txt"
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
           "TEST_TRACE": trace.as_posix(), "TEST_INSTALLED": (tmp_path / "installed").as_posix(), "TEST_TEMPLATE": template.as_posix()}
    return root, env, trace, template


def launch(project, **flags):
    root, env, _, _ = project
    return subprocess.run([SHELL, str(root / "start.sh")], cwd=root.parent, env={**env, **flags}, text=True, capture_output=True, timeout=30)


def test_start_sh_has_lf_and_valid_syntax():
    data = (ROOT / "start.sh").read_bytes()
    assert data.startswith(b"#!/bin/sh\n") and b"\r" not in data
    result = subprocess.run([SHELL, "-n", str(ROOT / "start.sh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fresh_environment_and_dependency_install(project):
    result = launch(project, TEST_MISSING_DEPS="1")
    assert result.returncode == 0, result.stderr
    trace = project[2].read_text(encoding="utf-8")
    assert "-m venv" in trace and "-m pip install -r" in trace
    assert str(project[0] / "app.py").replace("\\", "/").split(":")[-1] in trace


def test_existing_environment_is_reused_and_exit_code_preserved(project):
    root, _, trace, template = project
    (root / ".venv" / "bin").mkdir(parents=True)
    shutil.copyfile(template, root / ".venv" / "bin" / "python")
    (root / ".venv" / "bin" / "python").chmod(0o755)
    result = launch(project, TEST_APP_EXIT="7")
    assert result.returncode == 7, result.stderr
    calls = trace.read_text(encoding="utf-8")
    assert "-m venv" not in calls and "-m pip" not in calls


def test_foreign_or_broken_environment_is_not_removed(project):
    marker = project[0] / ".venv" / "Scripts" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")
    result = launch(project)
    assert result.returncode == 1 and "has not been deleted" in result.stderr
    assert marker.read_text(encoding="utf-8") == "keep"


def test_missing_supported_python_fails_clearly(project):
    result = launch(project, TEST_OLD="1")
    assert result.returncode == 1 and "Python 3.10+ was not found" in result.stderr
    assert not (project[0] / ".venv").exists()


def test_failed_dependency_install_does_not_start_server(project):
    result = launch(project, TEST_MISSING_DEPS="1", TEST_PIP_FAIL="1")
    assert result.returncode == 1 and "Dependency installation failed" in result.stderr
    assert "app.py" not in project[2].read_text(encoding="utf-8")
