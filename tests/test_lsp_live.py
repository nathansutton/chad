"""LIVE language-server tests: chad's client against a real pyright (and tsserver
when npx is present). Guarded, not mocked — this is the first test that exercises
the precision path end to end.

Gating: pyright resolves via PATH or uvx (`uvx chad-code` guarantees uv exists, so
this is exactly the shipped path). Without a resolvable launcher the tests skip;
with CHAD_LSP_LIVE_REQUIRED=1 (CI) a startup failure FAILS instead of skipping —
graceful degradation is for users, not for CI.
"""

import os
import shutil

import pytest

from chad import lspservers
from chad.lsp import LspService

_REQUIRED = os.environ.get("CHAD_LSP_LIVE_REQUIRED", "") == "1"


def _skip_or_fail(reason):
    if _REQUIRED:
        pytest.fail(f"live LSP required but unavailable: {reason}")
    pytest.skip(reason)


def _service_for(tmp_path, files, lang="python"):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    spec = lspservers.spec_for(lang)
    if lspservers.resolve_command(spec) is None:
        _skip_or_fail(f"no launcher for {lang} ({spec.tier})")
    svc = LspService(str(tmp_path))
    return svc


PY_LIB = "def helper(x):\n    return x + 1\n\n\ndef helper_twin(x):\n    return x\n"
PY_APP = "from lib import helper\n\n\ndef run():\n    return helper(1)\n"


@pytest.fixture
def py_service(tmp_path):
    svc = _service_for(tmp_path, {"lib.py": PY_LIB, "app.py": PY_APP})
    yield svc
    svc.stop()


def test_pyright_cross_file_references(py_service):
    # helper's identifier sits at (0-based) row 0 col 4 in lib.py
    locs = py_service.references("lib.py", 0, 4)
    if locs is None:
        _skip_or_fail("pyright failed to start")
    assert ("app.py", 5, 12) in locs or any(rel == "app.py" for rel, _l, _c in locs), locs
    # precision, not name-match: helper_twin's references stay empty
    twin = py_service.references("lib.py", 4, 4)
    assert twin == [], twin


def test_pyright_definition_and_hover(py_service):
    # `helper` used in app.py:5 resolves to its definition in lib.py:1
    defs = py_service.definition("app.py", 4, 11)
    if defs is None:
        _skip_or_fail("pyright failed to start")
    assert ("lib.py", 1) in defs, defs
    hover = py_service.hover("lib.py", 0, 4)
    assert hover and "helper" in hover, hover


def test_pyright_diagnostics_after_edit(py_service, tmp_path):
    (tmp_path / "bad.py").write_text("def f() -> int:\n    return 'nope'\n")
    if py_service.references("lib.py", 0, 4) is None:   # warm the server
        _skip_or_fail("pyright failed to start")
    diags = py_service.diagnostics_after_edit("bad.py")
    assert diags, "expected a type error from pyright"
    sev, line, msg = diags[0]
    assert sev == "error" and line == 2, diags

    # and a clean file gets a clean bill (empty list, not None)
    (tmp_path / "ok.py").write_text("def g() -> int:\n    return 3\n")
    assert py_service.diagnostics_after_edit("ok.py") == []


TS_LIB = "export function helper(x: number): number {\n  return x + 1;\n}\n"
TS_APP = 'import { helper } from "./lib";\n\nexport const r = helper(1);\n'


def test_typescript_cross_file_refs_and_pull_diagnostics(tmp_path):
    if shutil.which("typescript-language-server") is None and shutil.which("npx") is None:
        _skip_or_fail("no typescript-language-server or npx")
    svc = _service_for(tmp_path,
                       {"lib.ts": TS_LIB, "app.ts": TS_APP,
                        # the registry's honesty gate requires a project file
                        "tsconfig.json": '{"compilerOptions": {"strict": true}}\n'},
                       lang="typescript")
    try:
        locs = svc.references("lib.ts", 0, 16)
        if locs is None:
            _skip_or_fail("typescript server failed to start")
        assert any(rel == "app.ts" for rel, _l, _c in locs), locs
        # TS7's native LSP has no push diagnostics — this exercises the pull path
        (tmp_path / "bad.ts").write_text("export const n: number = 'nope';\n")
        diags = svc.diagnostics_after_edit("bad.ts")
        assert diags and diags[0][0] == "error", diags
    finally:
        svc.stop()
