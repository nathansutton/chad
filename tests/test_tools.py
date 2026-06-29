"""Characterization battery for the filesystem tools in `tools.py`:
`tool_edit` (uniqueness/no-corruption truth table), `tool_grep`, `tool_glob`,
`tool_write`. Pure filesystem in a temp dir — no model load.

The KEY invariant: `tool_edit` rewrites the file ONLY on a genuinely-unique match and
leaves it BYTE-FOR-BYTE untouched in every reject/ambiguous case. `test_edit.py` already
covers the recovery cascade (literal \\n, indent drift, ws-flexible ambiguity); this file
focuses on the plan's grep/glob/write tables plus the core edit truth table.

Run: `.venv/bin/python test_tools.py`
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chad import tools  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def _seed(files):
    """Create a fresh temp dir, write `files` (name->content, nested ok), chdir in.
    Caller MUST restore cwd in a finally."""
    d = tempfile.mkdtemp(prefix="tools_")
    for fn, c in files.items():
        p = os.path.join(d, fn)
        if os.path.dirname(fn):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(c.encode())
    os.chdir(d)
    return d


def _rawbytes(path):
    with open(path, "rb") as f:
        return f.read()


# --- tool_edit truth table ----------------------------------------------------

def test_edit_truth_table():
    cwd = os.getcwd()
    try:
        _seed({"f.py": "alpha\nbeta\ngamma\n"})

        # missing file
        check("edit: missing file",
              tools.tool_edit("nope.py", "a", "b").startswith("[no such file"))

        # old == new -> no-op, file untouched
        before = _rawbytes("f.py")
        res = tools.tool_edit("f.py", "beta", "beta")
        check("edit: no-op old==new", res.startswith("[no-op edit"), res)
        check("edit: no-op leaves file byte-identical", _rawbytes("f.py") == before)

        # old absent (and not whitespace-recoverable) -> not found, file untouched
        before = _rawbytes("f.py")
        res = tools.tool_edit("f.py", "qqq_absent_zzz", "whatever")
        check("edit: not found", res.startswith("[old string not found"), res)
        check("edit: not-found leaves file byte-identical", _rawbytes("f.py") == before)

        # old appears twice exactly -> ambiguous, file untouched
        _seed({"d.py": "x = 1\nx = 1\n"})
        before = _rawbytes("d.py")
        res = tools.tool_edit("d.py", "x = 1", "x = 9")
        check("edit: ambiguous n>1", res.startswith("[old string appears 2 times"), res)
        check("edit: ambiguous leaves file byte-identical", _rawbytes("d.py") == before)

        # unique match -> edited, file ACTUALLY rewritten
        _seed({"u.py": "alpha\nbeta\ngamma\n"})
        res = tools.tool_edit("u.py", "beta", "BETA")
        check("edit: unique returns [edited", res.startswith("[edited"), res)
        after = _rawbytes("u.py").decode()
        check("edit: unique applied the write", "BETA" in after and "beta" not in after, repr(after))
    finally:
        os.chdir(cwd)


# --- tool_grep ----------------------------------------------------------------

def test_grep():
    cwd = os.getcwd()
    try:
        _seed({
            "a.py": "import os\nNEEDLE here\nbye\n",
            "b.txt": "no match in here\n",
            "__pycache__/c.py": "NEEDLE in a skip dir\n",
        })

        # bad regex
        check("grep: bad regex", tools.tool_grep("(").startswith("[bad regex"))

        # a hit -> path:line: form
        out = tools.tool_grep("NEEDLE")
        check("grep: hit has path:line: form", "a.py:2:" in out, out)
        check("grep: hit shows the line text", "NEEDLE here" in out, out)

        # _SKIP_DIRS pruning: the __pycache__ copy must NOT appear
        check("grep: skip-dir pruned", "__pycache__" not in out, out)

        # no matches
        check("grep: no matches",
              tools.tool_grep("ZZZ_no_such_token") == "[no matches]")
    finally:
        os.chdir(cwd)


# --- tool_glob ----------------------------------------------------------------

def test_glob():
    cwd = os.getcwd()
    try:
        _seed({
            "b.py": "x\n",
            "a.py": "x\n",
            "__pycache__/c.py": "x\n",
            ".venv/d.py": "x\n",
        })

        # sorted top-level .py
        out = tools.tool_glob("*.py")
        check("glob: sorted .py", out == "a.py\nb.py", out)

        # recursive glob prunes skip dirs
        rec = tools.tool_glob("**/*.py")
        check("glob: skip-dir pruned",
              "__pycache__" not in rec and ".venv" not in rec, rec)
        check("glob: recursive still finds top-level",
              "a.py" in rec and "b.py" in rec, rec)

        # no matches
        check("glob: no matches", tools.tool_glob("*.nosuchext") == "[no matches]")
    finally:
        os.chdir(cwd)


# --- tool_write ---------------------------------------------------------------

def test_write():
    cwd = os.getcwd()
    try:
        _seed({})
        res = tools.tool_write("sub/dir/x.txt", "hello")
        check("write: [wrote N bytes]", res.startswith("[wrote 5 bytes"), res)
        check("write: created parent dirs + file", os.path.isfile("sub/dir/x.txt"))
        with open("sub/dir/x.txt") as f:
            check("write: file has the content", f.read() == "hello")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    test_edit_truth_table()
    test_grep()
    test_glob()
    test_write()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
