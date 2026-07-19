"""Micro-eval for the robust `tool_edit` recovery cascade (tools.py).

Each case is a real weak-model edit miss from the dogfooding logs (~1 in 6 edits failed:
literal "\\n" in `old`, indentation drift) or a SAFETY case (an ambiguous or wrong `old`
must NOT change the file). Pure + fast — no model load.

Run: `uv run python test_edit.py`
"""

import os
import tempfile

from chad.tools import tool_edit

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def run(before, old, new):
    """Apply tool_edit to a temp file; return (result_string, file_contents_after)."""
    d = tempfile.mkdtemp(prefix="edit_")
    p = os.path.join(d, "f.py")
    with open(p, "w") as f:
        f.write(before)
    res = tool_edit(p, old, new)
    with open(p) as f:
        after = f.read()
    return res, after


GEO = "def area(w, h):\n    return w * h\n"


def test_edit():
    # 1) exact match — fast path
    res, after = run(GEO, "    return w * h", "    return w * h * 2")
    check("exact applies", res.startswith("[edited") and "w * h * 2" in after, res)

    # 2) THE logged #1 failure: literal \n in `old` (geo.py loop), real newline in file
    res, after = run(GEO, "def area(w, h):\\n    return w * h",
                     "def area(w, h):\n    return w * h\n\ndef volume(w, h, d):\n    return w * h * d")
    check("literal-\\n old recovers", res.startswith("[edited") and "volume" in after, res)
    check("literal-\\n note explains", "escapes" in res, res)

    # 2b) literal \n in BOTH old and new (no real newline in new) -> both unescaped
    res, after = run(GEO, "def area(w, h):\\n    return w * h",
                     "def area(w, h):\\n    return w * w")
    check("literal-\\n both recovers", res.startswith("[edited"), res)
    check("new unescaped to real newline", "\\n" not in after and "w * w" in after, repr(after))
    # 2c) item 4: unescaping `new` is ambiguous (did the model mean a newline
    # or a literal backslash-n?), so when we DO transform it the result must DISCLOSE it
    # rather than change the model's replacement silently.
    check("new-unescape disclosed", "interpreted as newline" in res, res)

    # 2d) item 4 SAFETY: a literal backslash-n in `new` on the EXACT-match path is written
    # verbatim (no silent newline transform) and no disclosure fires — a model that wants a
    # real "\\n" in its source gets exactly that.
    res, after = run("x = 1\n", "x = 1", "x = 1  # keep \\n literal")
    check("literal \\n in new preserved on exact path",
          "\\n" in after and "interpreted" not in res, repr((res, after)))

    # 3) indentation drift: model quoted with no leading indent (construct.py case)
    src = "def f(text):\n        return text.strip()\n"
    res, after = run(src, "return text.strip()", "return text.split('X')[-1].strip()")
    check("indent-drift recovers", res.startswith("[edited"), res)
    check("indent preserved on recovery", "        return text.split" in after, repr(after))

    # 3b) trailing-whitespace drift
    src = "x = 1   \ny = 2\n"
    res, after = run(src, "x = 1", "x = 99")
    check("trailing-ws recovers", res.startswith("[edited") and "x = 99" in after, res)

    # --- SAFETY: these must NOT change the file -----------------------------
    # 4) genuinely absent string -> no edit + a close-match hint
    res, after = run(GEO, "return w / h", "return w // h")
    check("absent: not applied", "not found" in res and after == GEO, res)
    check("absent: gives closest-line hint", "Closest line" in res, res)

    # 5) exact-duplicate old -> ambiguous, no edit
    dup = "a = 1\nb = 2\na = 1\n"
    res, after = run(dup, "a = 1", "a = 7")
    check("exact dup: not applied", "appears 2 times" in res and after == dup, res)

    # 6) whitespace-flexible AMBIGUOUS: a tab-indented `old` is no exact substring of
    # the space-indented file, but strips to two matching lines -> refuse, no edit.
    dup2 = "    return x\nfoo\n    return x\n"
    res, after = run(dup2, "\treturn x", "return y")
    check("ws-flexible dup: not applied", "matches 2 places" in res and after == dup2, res)

    # 7) no-op (old == new)
    res, after = run(GEO, "    return w * h", "    return w * h")
    check("no-op rejected", "no-op" in res and after == GEO, res)


def test_ws_recovery_prefers_file_indentation():
    """Iter-2 (sphinx-7440): on the whitespace-flexible path the model's
    RELATIVE indents are the least trustworthy part of the edit. A same-line-count
    replacement takes each replaced line's indent from the FILE, so a garbled-indent
    `new` can no longer land a SyntaxError."""
    before = "def f():\n    a = 1\n    b = compute(a.lower(), x)\n"
    # `old` drifts in indentation (ws-flex match); `new` carries broken relative
    # indents (2 then 10) — exactly the 7440 corruption. File indents must win.
    res, after = run(before,
                     "  a = 1\n        b = compute(a.lower(), x)",
                     "  a = 1\n          b = compute(a, x)")
    check("ws-recovery: edit landed", res.startswith("[edited"), res)
    check("ws-recovery: file indentation kept",
          after == "def f():\n    a = 1\n    b = compute(a, x)\n", repr(after))


def test_ws_only_edit_applies_verbatim():
    """Iter-2 (sphinx-7440): an indentation-ONLY fix used to normalize to
    '[no-op edit]' — a broken indent was literally unrepairable through this tool
    and the model fell back to blind sed. When reindenting reproduces the file
    byte-for-byte but the model's `new` differs, trust its whitespace verbatim."""
    before = "def f():\n    a = 1\n          b = 2\n"   # broken indent on b
    res, after = run(before, "a = 1\n b = 2", "    a = 1\n    b = 2")
    check("ws-only: applied verbatim", "verbatim" in res, res)
    check("ws-only: indent repaired",
          after == "def f():\n    a = 1\n    b = 2\n", repr(after))


def test_indent_break_is_rejected_and_reverted():
    """Prong 1: an edit that would newly introduce a Python IndentationError
    (or TabError) to a CLEAN file is rejected and the file left untouched — so the model
    re-sends the edit instead of looping on whitespace surgery to un-break a landed file."""
    before = "def f():\n    a = 1\n    b = 2\n"

    # over-indent mid-block -> IndentationError ("unexpected indent")
    res, after = run(before, "    b = 2", "      b = 2")
    check("indent-break rejected", res.startswith("[edit rejected"), res)
    check("indent-break file untouched", after == before, repr(after))

    # spaces -> tab in a space-indented block -> TabError (a subclass of IndentationError)
    res, after = run(before, "    b = 2", "\tb = 2")
    check("tab-break rejected", res.startswith("[edit rejected"), res)
    check("tab-break file untouched", after == before, repr(after))


def test_non_indent_break_rejected(monkeypatch):
    """(supersedes the warn-only scope): a NON-indentation syntax
    break (unclosed paren) is also REVERTED. The 073 dogfood measured what warn-only
    costs a small model: ten ignored "no longer parses" warnings while stale line edits
    compounded on a severed def signature, ending in LOOP ABORT with the file broken.
    Multi-step changes that must pass through a broken state route through `write`
    (still warn-only) or replace_symbol. Ablating `syntax_revert` restores warn-and-land."""
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    res, after = run("x = 1\n", "x = 1", "x = (1")
    check("non-indent break rejected", res.startswith("[edit rejected"), res)
    check("file left unchanged", after == "x = 1\n", repr(after))
    monkeypatch.setenv("CHAD_DISABLE", "syntax_revert")
    res, after = run("x = 1\n", "x = 1", "x = (1")
    check("ablated: break lands with warning",
          res.startswith("[edited") and "no longer parses" in res and "x = (1" in after, res)


def test_already_broken_file_stays_editable():
    """Prong 1 boundary: when `before` is ALREADY broken, indent_reject stays out of the
    way (parse of `before` fails), so a fix that passes through a still-broken state is
    never stranded — the sphinx-7440 repair path keeps working."""
    before = "def f():\n    a = 1\n      b = 2\n"      # b over-indented: already broken
    res, after = run(before, "    a = 1", "    a = 111")  # edit unrelated line
    check("broken-file edit lands", res.startswith("[edited") and "a = 111" in after, res)


def test_failed_edit_shows_visible_whitespace():
    """Prong 2: a no-op / not-found edit hands back the target lines with
    leading whitespace made visible (· space, → tab), so the model copies the exact
    indentation instead of re-guessing the column count."""
    # no-op (old == new): still echo the current line with visible indentation
    res, _ = run(GEO, "    return w * h", "    return w * h")
    check("no-op shows visible ws", "·" in res and "no-op" in res, res)

    # not-found close match: show the closest line's real indentation
    res, _ = run(GEO, "return w / h", "return w // h")
    check("not-found shows visible ws", "·" in res, res)


TAB_OBJ = (
    "export const keys = {\n"
    "\tjson: true,\n"
    "\ttimeout: true,\n"
    "};\n"
)


def test_ws_recovery_inserted_line_takes_neighbor_indent():
    # ky-timeoutMessage (session dbf9dee0/20260713): model's old/new carry wrong
    # absolute AND relative tabs; the inserted line must inherit its resolved
    # neighbor's file indent, not first-line math.
    old = "\texport const keys = {\n\t\tjson: true,\n\t\ttimeout: true,\n\t};"
    new = ("\texport const keys = {\n\t\tjson: true,\n\t\ttimeout: true,\n"
           "\t\t\ttimeoutMessage: true,\n\t};")
    res, after = run(TAB_OBJ, old, new)
    check("insert: recovered", "recovered" in res, res)
    check("insert: neighbor tab depth",
          "\n\ttimeoutMessage: true,\n" in after, repr(after))


def test_ws_recovery_insert_after_opener_indents_one_unit():
    old = "\texport const keys = {\n\t\tjson: true,"
    new = "\texport const keys = {\n\t\t\tfirst: true,\n\t\tjson: true,"
    res, after = run(TAB_OBJ, old, new)
    check("opener: one unit deeper", "\n\tfirst: true,\n\tjson: true,\n" in after,
          repr(after))


def test_recovery_result_echoes_landed_indentation():
    old = "\texport const keys = {\n\t\tjson: true,\n\t\ttimeout: true,\n\t};"
    new = ("\texport const keys = {\n\t\tjson: true,\n\t\ttimeout: true,\n"
           "\t\t\ttimeoutMessage: true,\n\t};")
    res, _ = run(TAB_OBJ, old, new)
    check("echo: legend present", "→ = one tab" in res, res)
    check("echo: shows landed tabs", "→timeoutMessage: true," in res, res)


def test_ws_only_edit_result_echoes_landed_indentation():
    # `old` strips to the file lines but is NOT an exact substring (no leading tab), so
    # this exercises the ws-flexible verbatim path — not the exact-match fast path, which
    # the plan keeps echo-free. ('s literal `old` accidentally exact-matched.)
    res, _ = run(TAB_OBJ, "json: true,\ntimeout: true,",
                 "\t\tjson: true,\n\t\ttimeout: true,")
    check("ws-only echo", "→→json: true," in res, res)


if __name__ == "__main__":
    test_edit()
    test_ws_recovery_prefers_file_indentation()
    test_ws_only_edit_applies_verbatim()
    test_indent_break_is_rejected_and_reverted()
    test_already_broken_file_stays_editable()
    test_failed_edit_shows_visible_whitespace()
    test_ws_recovery_inserted_line_takes_neighbor_indent()
    test_ws_recovery_insert_after_opener_indents_one_unit()
    test_recovery_result_echoes_landed_indentation()
    test_ws_only_edit_result_echoes_landed_indentation()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
