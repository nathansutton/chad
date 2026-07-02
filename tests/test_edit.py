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
    # 2c) item 4 (plan 044): unescaping `new` is ambiguous (did the model mean a newline
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


if __name__ == "__main__":
    test_edit()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
