"""Characterization battery for the filesystem tools in `tools.py`:
`tool_edit` (uniqueness/no-corruption truth table) and `tool_write`. Pure filesystem
in a temp dir — no model load.

The KEY invariant: `tool_edit` rewrites the file ONLY on a genuinely-unique match and
leaves it BYTE-FOR-BYTE untouched in every reject/ambiguous case. `test_edit.py` already
covers the recovery cascade (literal \\n, indent drift, ws-flexible ambiguity); this file
focuses on the write table plus the core edit truth table.

Run: `.venv/bin/python test_tools.py`
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chad import tools  # noqa: E402

PASS = 0
FAIL = 0


def _with_lever_off(name):
    """Run a test with one lever subtracted from the suite's CHAD_ENABLE=all — for
    tests that pin the legacy (lever-OFF) output shape while the rest of the suite
    exercises everything ON."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            old = os.environ.get("CHAD_DISABLE")
            os.environ["CHAD_DISABLE"] = f"{old},{name}" if old else name
            try:
                return fn(*a, **k)
            finally:
                if old is None:
                    os.environ.pop("CHAD_DISABLE", None)
                else:
                    os.environ["CHAD_DISABLE"] = old
        return wrapper
    return deco


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


# --- tool_bash ----------------------------------------------------------------

# Line clipping ablated: every oversized fixture here is one 40k-char blob, and
# this test pins the head/tail budget's shape on it. test_bash_line_clip covers the
# clip path, including that a blob is what it exists for.
@_with_lever_off("bash_line_clip")
def test_bash():
    # short output passes through untouched
    check("bash: short echo", tools.tool_bash("printf hi") == "hi")

    # no output -> sentinel
    check("bash: no output", tools.tool_bash("true") == "[no output]")

    # nonzero exit -> [exit N] prefix
    res = tools.tool_bash("printf oops; exit 3")
    check("bash: exit prefix", res.startswith("[exit 3]\n") and "oops" in res, res)

    # The timeout-KILL path: timeout with no output before the kill -> bare
    # sentinel (0s deadline fires immediately, before `sleep` prints anything).
    check("bash: timeout no output", tools.tool_bash("sleep 5", timeout=0)
          == "[timed out after 0s; no output before it was killed]")

    # timeout WITH partial output -> the output the process printed before the kill
    # is preserved (the whole point: a killed build/train still printed how far it
    # got). Print a marker, then sleep past a short deadline so the kill lands
    # mid-run.
    partial = tools.tool_bash("printf 'PROGRESS_50_PERCENT\\n'; sleep 5", timeout=1)
    check("bash: timeout keeps partial", "PROGRESS_50_PERCENT" in partial, partial)
    check("bash: timeout names the kill", partial.startswith("[timed out after 1s;"),
          partial[:60])

    # interrupt (should_stop) also preserves partial output
    stop = {"n": 0}
    def should_stop():
        stop["n"] += 1
        return stop["n"] > 3  # let a couple poll cycles pass so the printf lands first
    intr = tools.tool_bash("printf 'PARTIAL_BEFORE_CTRLC\\n'; sleep 5", should_stop=should_stop)
    check("bash: interrupt keeps partial", "PARTIAL_BEFORE_CTRLC" in intr, intr)
    check("bash: interrupt names the stop", intr.startswith("[interrupted by user;"), intr[:60])

    # long output keeps HEAD + TAIL + an omission marker, and the tail bias means
    # the last line (a failure summary lives here) survives when a head-only cut
    # would drop it.
    big = tools.tool_bash(
        "printf 'FIRST_LINE\\n'; head -c 40000 /dev/zero | tr '\\0' 'x'; "
        "printf '\\nLAST_LINE_FAILURE_SUMMARY\\n'")
    check("bash: long output truncated", len(big) < 25000, len(big))
    check("bash: keeps head", "FIRST_LINE" in big, big[:80])
    check("bash: keeps tail", "LAST_LINE_FAILURE_SUMMARY" in big, big[-80:])
    check("bash: omission marker", "chars omitted" in big, big)

    # the [exit N] prefix survives truncation (it's at the head we keep)
    fail = tools.tool_bash(
        "printf 'HEAD\\n'; head -c 40000 /dev/zero | tr '\\0' 'x'; "
        "printf '\\nTAIL\\n'; exit 1")
    check("bash: exit prefix survives truncation", fail.startswith("[exit 1]\n"), fail[:40])


def _spill_path_from(result):
    """Extract the spill path the truncation notice points at."""
    return result.split("FULL output saved to ", 1)[1].split(";", 1)[0]


@_with_lever_off("bash_trim_keep_failures")
def test_bash_line_clip():
    """One absurdly long line is capped instead of eating the whole budget, and the
    full text stays recoverable. The measured case: a search that reached a gitignored
    build directory returned a 92k-char .js.map line, and head+tail handed back 20k
    chars sliced out of the middle of base64 — ~5k tokens of prefill, no information."""
    old = os.environ.get("CHAD_SPILL_DIR")
    d = tempfile.mkdtemp(prefix="clip_")
    os.environ["CHAD_SPILL_DIR"] = d
    try:
        blob = "M" * 92000
        out = tools._bash_headtail(f"head.js.map:1:{blob}\nreal line\n")
        check("clip: result is small", len(out) < 1200, len(out))
        check("clip: keeps the line's start", out.startswith("head.js.map:1:MMM"), out[:40])
        check("clip: marks the line", "…[line clipped]" in out, out[:800])
        check("clip: other lines intact", "real line" in out, out)
        check("clip: names the amount", "chars clipped from over-long lines" in out, out)
        with open(_spill_path_from(out)) as f:
            full = f.read()
        check("clip: spill holds the UNCLIPPED original", blob in full, len(full))

        # a small clip is not worth a file; the notice just says what happened
        small = tools._bash_headtail("x" * (tools.BASH_LINE_CHARS + 50) + "\n")
        check("clip: small clip has no spill", "saved to" not in small, small[-120:])
        check("clip: small clip still marked", "…[line clipped]" in small, small[-120:])

        # normal line-shaped output is byte-identical
        plain = "line one\nline two\n"
        check("clip: normal output untouched", tools._bash_headtail(plain) == plain)
    finally:
        if old is None:
            os.environ.pop("CHAD_SPILL_DIR", None)
        else:
            os.environ["CHAD_SPILL_DIR"] = old


@_with_lever_off("bash_line_clip")  # single-blob fixtures; see test_bash
def test_bash_spill():
    """Truncation spills the FULL output to a session-scoped file: the
    omitted middle is a `grep <path>` away instead of a full re-run away."""
    old = os.environ.get("CHAD_SPILL_DIR")
    d = tempfile.mkdtemp(prefix="spill_")
    os.environ["CHAD_SPILL_DIR"] = d
    session_dir = os.path.join(d, str(os.getpid()))
    try:
        # oversized output -> notice names an absolute, existing path whose file holds
        # the COMPLETE original output. The marker sits at ~char 20k — inside the
        # omitted middle (head keeps 8k, tail keeps the last 12k of ~40k) — so it is
        # recoverable ONLY through the spill file.
        big = tools.tool_bash(
            "head -c 20000 /dev/zero | tr '\\0' 'x'; printf 'MIDDLE_MARKER'; "
            "head -c 20000 /dev/zero | tr '\\0' 'y'")
        check("spill: notice present", "FULL output saved to " in big, big)
        path = _spill_path_from(big)
        check("spill: path absolute", os.path.isabs(path), path)
        check("spill: file exists", os.path.exists(path), path)
        check("spill: file is 0600", os.stat(path).st_mode & 0o777 == 0o600,
              oct(os.stat(path).st_mode))
        with open(path) as f:
            full = f.read()
        check("spill: marker omitted from transcript", "MIDDLE_MARKER" not in big)
        check("spill: middle recoverable from file", "MIDDLE_MARKER" in full)
        check("spill: file holds the complete output", len(full) == 40013, len(full))

        # output under the cap -> no new spill file
        before = len(os.listdir(session_dir))
        check("spill: small output untouched", tools.tool_bash("printf hi") == "hi")
        check("spill: small output makes no file",
              len(os.listdir(session_dir)) == before, os.listdir(session_dir))

        # killed command with a large partial -> same spill, and the [timed out
        # prefix guardrails.py keys on stays FIRST (the notice lives mid-string).
        killed = tools.tool_bash(
            "head -c 15000 /dev/zero | tr '\\0' 'x'; printf 'KILLED_MID'; "
            "head -c 15000 /dev/zero | tr '\\0' 'y'; sleep 5", timeout=1)
        check("spill: killed keeps prefix", killed.startswith("[timed out after 1s;"),
              killed[:60])
        check("spill: killed gets a path", "FULL output saved to " in killed, killed[:400])
        with open(_spill_path_from(killed)) as f:
            check("spill: killed middle recoverable", "KILLED_MID" in f.read())

        # cap: 25 more oversized results -> only the newest BASH_SPILL_KEEP files remain
        for i in range(25):
            tools._bash_headtail(f"run{i}-" + "z" * tools.BASH_MAX_CHARS)
        names = [n for n in os.listdir(session_dir)
                 if n.startswith("bash-") and n.endswith(".log")]
        check("spill: capped per session", len(names) == tools.BASH_SPILL_KEEP, len(names))

        # stale sibling session dirs get swept on the first spill of a process
        stale = os.path.join(d, "99999999")
        os.makedirs(stale)
        os.utime(stale, (0, 0))
        tools.spill._SWEPT = False
        tools._bash_headtail("s" * (tools.BASH_MAX_CHARS + 1))
        check("spill: stale session swept", not os.path.exists(stale))

        # a failed spill degrades to the old notice, never breaks the result
        os.environ["CHAD_SPILL_DIR"] = "/dev/null/nope"
        deg = tools._bash_headtail("q" * (tools.BASH_MAX_CHARS + 1))
        check("spill: failure falls back", "output truncated" in deg
              and "saved to" not in deg, deg[7990:8100])
    finally:
        if old is None:
            os.environ.pop("CHAD_SPILL_DIR", None)
        else:
            os.environ["CHAD_SPILL_DIR"] = old


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
