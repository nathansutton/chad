"""Pin lsp.py's import-degradation contract (plan 049).

The precision layer ships in the optional `lsp` extra (serena-agent). These two
tests pin what a default install must guarantee: extension routing works, and a
missing solidlsp package degrades to "unavailable" (callers fall back to
tree-sitter) instead of raising. The full lsp.py backfill (server-backed) is a
separate plan.
"""

import sys

from chad.lsp import LspService, lang_for


def test_lang_for_known_and_unknown():
    assert lang_for("x.py") == "PYTHON"
    assert lang_for("sub/dir/x.PY") == "PYTHON"  # extension match is case-insensitive
    assert lang_for("x.unknownext") is None
    assert lang_for("no_extension") is None


def test_degrades_without_solidlsp(tmp_path, monkeypatch):
    # None in sys.modules makes `from solidlsp import ...` raise ImportError,
    # exactly like a default (no-extra) install where the package is absent.
    monkeypatch.setitem(sys.modules, "solidlsp", None)
    svc = LspService(str(tmp_path))
    assert svc.available("f.py") is False
    assert svc.references("f.py", 0, 0) is None
    assert svc.definition("f.py", 0, 0) is None
    svc.stop()  # no server was created; must be a no-op, not an error
