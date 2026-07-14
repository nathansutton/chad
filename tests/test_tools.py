"""Characterization battery for the filesystem tools in `tools.py`:
`tool_edit` (uniqueness/no-corruption truth table), `tool_grep`, `tool_glob`,
`tool_write`. Pure filesystem in a temp dir — no model load.

The KEY invariant: `tool_edit` rewrites the file ONLY on a genuinely-unique match and
leaves it BYTE-FOR-BYTE untouched in every reject/ambiguous case. `test_edit.py` already
covers the recovery cascade (literal \\n, indent drift, ws-flexible ambiguity); this file
focuses on the plan's grep/glob/write tables plus the core edit truth table.

Run: `.venv/bin/python test_tools.py`
"""

import glob as _glob
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


# --- tool_bash ----------------------------------------------------------------

def test_bash():
    # short output passes through untouched
    check("bash: short echo", tools.tool_bash("printf hi") == "hi")

    # no output -> sentinel
    check("bash: no output", tools.tool_bash("true") == "[no output]")

    # nonzero exit -> [exit N] prefix
    res = tools.tool_bash("printf oops; exit 3")
    check("bash: exit prefix", res.startswith("[exit 3]\n") and "oops" in res, res)

    # timeout with no output before the kill -> bare sentinel (0s deadline fires
    # immediately, before `sleep` prints anything)
    check("bash: timeout no output", tools.tool_bash("sleep 5", timeout=0)
          == "[timed out after 0s; no output before it was killed]")

    # timeout WITH partial output -> the output the process printed before the kill is
    # preserved (the whole point: a killed build/train still printed how far it got).
    # Print a marker, then sleep past a short deadline so the kill lands mid-run.
    partial = tools.tool_bash("printf 'PROGRESS_50_PERCENT\\n'; sleep 5", timeout=1)
    check("bash: timeout keeps partial", "PROGRESS_50_PERCENT" in partial, partial)
    check("bash: timeout names the kill", partial.startswith("[timed out after 1s;"), partial[:60])

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


def test_bash_spill():
    """Truncation spills the FULL output to a session-scoped file (plan 02): the
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
        # prefix guardrails.py keys on stays FIRST (the notice lives mid-string)
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
        tools._SPILL_SWEPT = False
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
              tools.tool_grep("ZZZ_no_such_token").startswith("[no matches for"))
    finally:
        os.chdir(cwd)


def test_grep_default_byte_identical():
    """Default-args output must stay byte-for-byte what the pre-plan-037 code emitted:
    `path:line: text` lines joined by \\n, no notices when no cap binds."""
    cwd = os.getcwd()
    try:
        _seed({"a.py": "import os\nNEEDLE here\nalso NEEDLE\nbye\n"})
        out = tools.tool_grep("NEEDLE")
        check("grep: default byte-identical",
              out == "./a.py:2: NEEDLE here\n./a.py:3: also NEEDLE", repr(out))
    finally:
        os.chdir(cwd)


def test_grep_line_cap():
    """A match inside a huge single line is clipped so it can't blow up the transcript."""
    cwd = os.getcwd()
    try:
        _seed({"big.js": "x" * 50000 + "NEEDLE" + "y" * 50000 + "\n"})
        out = tools.tool_grep("NEEDLE")
        check("grep: single output line", "\n" not in out, len(out))
        check("grep: line clipped to <=600", len(out) <= 600, len(out))
        check("grep: clip marker present", "…[line clipped]" in out, out)
    finally:
        os.chdir(cwd)


def test_grep_truncation_notices():
    """The 200-line output cap announces itself with shown/total counts."""
    cwd = os.getcwd()
    try:
        _seed({"m.txt": "".join(f"NEEDLE line {i}\n" for i in range(500))})
        out = tools.tool_grep("NEEDLE")
        lines = out.splitlines()
        check("grep: capped at 200 + notice", len(lines) == 201, len(lines))
        check("grep: truncation notice text",
              lines[-1] == "[results truncated: 200/500 lines — narrow the pattern "
                           "or add a path]", lines[-1])
    finally:
        os.chdir(cwd)


def test_grep_path_is_file():
    """A file passed as `path` is searched directly — the old dir-walk treatment made
    it silently match nothing, and the model passes file paths constantly."""
    cwd = os.getcwd()
    try:
        _seed({"a.py": "NEEDLE here\n", "b.py": "NEEDLE too\n"})
        out = tools.tool_grep("NEEDLE", path="a.py")
        check("grep: file path searches the file", "a.py:1:" in out, out)
        check("grep: file path scoped to that file", "b.py" not in out, out)

        # naming the file explicitly overrides the skip list, like `read` does
        _seed({"__pycache__/c.py": "NEEDLE in a skip dir\n"})
        out = tools.tool_grep("NEEDLE", path="__pycache__/c.py")
        check("grep: explicit file beats skip list", "c.py:1:" in out, out)
    finally:
        os.chdir(cwd)


def test_grep_path_not_found():
    """A nonexistent `path` announces itself instead of reading as a clean no-match."""
    cwd = os.getcwd()
    try:
        _seed({"a.py": "NEEDLE\n"})
        out = tools.tool_grep("NEEDLE", path="no/such/dir")
        check("grep: missing path is loud",
              out == "[path not found: no/such/dir]", out)
    finally:
        os.chdir(cwd)


def test_grep_dirs_do_not_starve_file_cap():
    """The GREP_MAX_FILES budget must count only files we'd actually SEARCH — not the
    directories (and skipped blobs) the walk passes through. The demonstrated starvation
    (django-16454): a dir-heavy tree exhausted the cap on directories before the walk
    reached the file holding the target symbol. Here we seed many empty subdirs ahead of
    the one real match and shrink the cap below the dir count: the match must still land,
    and truncation must NOT be reported (no real file was dropped)."""
    cwd = os.getcwd()
    saved = tools.GREP_MAX_FILES
    try:
        d = _seed({"hit.py": "NEEDLE lives here\n"})
        # Many EMPTY dirs the walk yields (before files, at each level) but that hold
        # nothing to search. There is exactly ONE real file, so walk order is irrelevant.
        for i in range(30):
            os.makedirs(os.path.join(d, f"empty{i:03d}"), exist_ok=True)
        tools.GREP_MAX_FILES = 3  # far below the dir count; before the fix, dirs ate it
        out = tools.tool_grep("NEEDLE")
        check("grep: dirs don't consume the file budget", "hit.py:1:" in out, out)
        check("grep: one real file is not reported truncated", "searched first" not in out, out)
    finally:
        tools.GREP_MAX_FILES = saved
        os.chdir(cwd)


def test_grep_ignore_case_and_context():
    cwd = os.getcwd()
    try:
        _seed({"a.py": "alpha\nBETA match\ngamma\ndelta\n"})

        # ignore_case: lowercase pattern finds the uppercase line
        ci = tools.tool_grep("beta", ignore_case=True)
        check("grep: ignore_case hits", "a.py:2:" in ci and "BETA match" in ci, ci)
        check("grep: case-sensitive misses",
              tools.tool_grep("beta").startswith("[no matches for"))

        # context: N lines before/after, match uses ':' and context uses '-'
        c1 = tools.tool_grep("BETA", context=1)
        check("grep: context before", "a.py:1- alpha" in c1, c1)
        check("grep: context match sep", "a.py:2: BETA match" in c1, c1)
        check("grep: context after", "a.py:3- gamma" in c1, c1)
        check("grep: no stray group sep for single group", "--" not in c1, c1)

        # two non-adjacent matches -> two groups separated by --
        _seed({"b.py": "one\nHIT\nx\nx\nx\nx\nHIT\nend\n"})
        c2 = tools.tool_grep("HIT", context=1)
        check("grep: two groups separated", "\n--\n" in c2, c2)

        # context clamps to 0-5 (6 is treated as 5, still valid output)
        check("grep: context clamp doesn't error",
              "b.py:" in tools.tool_grep("HIT", context=6))
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


def test_walk_fast_path_matches_glob():
    """The pruned-walk fast path (plan: profiling pass) must return the same set as
    glob for the basename-only patterns it claims, dirs included, dotfiles excluded;
    structured patterns must decline (None) so callers fall back."""
    cwd = os.getcwd()
    try:
        _seed({
            "a.py": "x\n", "sub/b.py": "x\n", "sub/deep/c.txt": "x\n",
            ".hidden/d.py": "x\n", ".dotfile.py": "x\n",
            "node_modules/e.py": "x\n",
        })
        for pat in ("**/*", "**/*.py", "**/b.py"):
            fast = sorted(tools._walk_glob(".", pat))
            # one deliberate difference: glob+_skip kept the ignored dir ITSELF as an
            # entry ("./node_modules" has no trailing slash for _skip to see); the
            # walker prunes it entirely, so drop those from the parity expectation
            slow = sorted(h for h in _glob.glob(os.path.join(".", pat), recursive=True)
                          if not tools._skip(h)
                          and os.path.basename(h) not in tools.IGNORE_DIRS)
            check(f"walk fast path == glob for {pat}", fast == slow, (pat, fast, slow))
        for pat in ("*.py", "sub/*.py", "sub/**/*.py", "**/deep/*.txt"):
            check(f"structured pattern {pat} falls back", tools._walk_glob(".", pat) is None)
    finally:
        os.chdir(cwd)


def test_grep_prescreen_edge_patterns():
    """Patterns the whole-file prescreen can't mirror (lookarounds, \\A/\\Z) must skip
    it and still match per-line; anchored patterns must survive the MULTILINE probe."""
    cwd = os.getcwd()
    try:
        _seed({"a.py": "foo here\nbar\nfoo\n"})
        check("negative lookahead still matches",
              "a.py:3:" in tools.tool_grep(r"foo(?! here)"), tools.tool_grep(r"foo(?! here)"))
        check("^ anchor matches mid-file", "a.py:2:" in tools.tool_grep(r"^bar$"))
        check(r"\A pattern skips prescreen, matches line 1",
              "a.py:1:" in tools.tool_grep(r"\Afoo"))
    finally:
        os.chdir(cwd)


def test_grep_big_file_streams():
    """Files over GREP_FULLREAD_MAX skip the prescreen read and stream line-by-line."""
    cwd = os.getcwd()
    try:
        _seed({"big.txt": "pad\n" * (tools.GREP_FULLREAD_MAX // 4) + "NEEDLE end\n"})
        out = tools.tool_grep("NEEDLE")
        check("grep: match found past the full-read cap", "NEEDLE end" in out, out[:120])
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
    test_bash()
    test_grep()
    test_grep_default_byte_identical()
    test_grep_line_cap()
    test_grep_truncation_notices()
    test_grep_path_is_file()
    test_grep_path_not_found()
    test_grep_dirs_do_not_starve_file_cap()
    test_grep_ignore_case_and_context()
    test_glob()
    test_write()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
