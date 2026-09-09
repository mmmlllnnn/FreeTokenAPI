from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "freetokenapi"
ENTRY_SCRIPTS = [ROOT / "app.py"]


def test_package_and_entry_scripts_parse_as_python_3_10():
    errors = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")) + ENTRY_SCRIPTS:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)}: line {exc.lineno}: {exc.msg}")
    assert not errors, "\n".join(errors)
