"""Spill files: the disk half of every truncation chad performs.

One rule, applied everywhere the harness removes bytes from the transcript: a clip
is a loan, not a deletion. The dropped body is written here first and the truncation
notice names the path, so the model's way back to it is `grep <path>` — seconds —
rather than re-running the command or re-reading the file, which costs a whole turn
plus a re-prefill on a local model.

That rule is measured, not aesthetic. Over the banked session archive the results
that compaction head/tail-trimmed were later re-fetched with a byte-identical call
7.6% of the time (17 of 225, in 7 sessions); each one is a dead turn at 30-60 s.
The per-result clip, whose notice already told the model to narrow its query, was
re-run 0 times out of 21 — the difference is whether the notice hands back a
pointer or an instruction.

Bounded three ways, because chad has already had one unbounded-cache disk incident:
per-kind file counts (`KEEP`) inside a per-process session dir, a dir-wide byte
budget (`MAX_DIR_BYTES`), and a sweep of session dirs orphaned by dead processes.
Every write is best-effort — a disk error degrades the notice, never the result.
"""

import itertools
import os
import re
import shutil
import time

# How many files of each kind a session dir keeps, newest-first. `bash` and `result`
# fire a handful of times per turn; a single compaction pass can trim dozens of tool
# results at once (measured: ~6 per compacted session, with a long tail), so `compact`
# gets a wider count. The byte budget below is the real backstop for all of them.
KEEP = {"bash": 20, "compact": 120, "result": 20}
# Total bytes of spill a single session may hold. Well under the ~50 MB/session line
# where the retention budget would need to be tightened rather than merely enforced.
MAX_DIR_BYTES = 32 * 1024 * 1024
_MAX_AGE_S = 7 * 24 * 3600

_IDS = itertools.count(1)
_SWEPT = False
_NAME_RE = re.compile(r"^([a-z]+)-(\d+)\.log$")
# Every truncation notice chad writes names its file the same way ("… saved to
# <path>"), which lets a result that has ALREADY been spilled be recognised and
# re-pointed instead of written to disk a second time — a bash result that overflows
# its own head/tail budget and then the generic per-result cap is one body, one file.
POINTER_RE = re.compile(r"saved to (/\S+?\.log)\b")


def base_dir() -> str:
    """Root under which every session's spill dir lives. Resolved fresh each call
    (the eval harness repoints HOME per task). Never under cwd: a stray file would
    pollute the project's git status and fail the plan-mode untouched-tree check."""
    return os.environ.get("CHAD_SPILL_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "chad", "spill")


def session_dir() -> str:
    """This process's spill dir. A spill is only useful to the session that wrote
    it, so the pid is the natural retention unit."""
    return os.path.join(base_dir(), str(os.getpid()))


def _sweep_stale(base: str) -> None:
    """Delete sibling session dirs untouched for _MAX_AGE_S — orphaned by a dead
    process. Best-effort, runs once per process (on the first spill). Never raises."""
    try:
        deadline = time.time() - _MAX_AGE_S
        me = str(os.getpid())
        for name in os.listdir(base):
            path = os.path.join(base, name)
            try:
                if name != me and os.path.isdir(path) and os.path.getmtime(path) < deadline:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass
    except OSError:
        pass


def _entries(d: str):
    """[(seq, kind, name)] for the spill files in `d`, oldest-first. The sequence
    number is monotonic within a process, so it orders across kinds too."""
    out = []
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for name in names:
        m = _NAME_RE.match(name)
        if m:
            out.append((int(m.group(2)), m.group(1), name))
    out.sort()
    return out


def _prune(d: str) -> None:
    """Enforce both budgets, oldest-first: the per-kind file count, then the
    dir-wide byte total. Never raises."""
    entries = _entries(d)
    doomed = []
    for kind in {k for _, k, _ in entries}:
        same = [e for e in entries if e[1] == kind]
        keep = KEEP.get(kind, 20)
        doomed.extend(same[:-keep] if keep else same)
    live = [e for e in entries if e not in doomed]
    total = 0
    sizes = {}
    for e in live:
        try:
            sizes[e] = os.path.getsize(os.path.join(d, e[2]))
        except OSError:
            sizes[e] = 0
        total += sizes[e]
    # Keep the newest file unconditionally: it is the one the notice just pointed at,
    # and a single oversized body must degrade the budget, not vanish from under its
    # own pointer.
    for e in live[:-1]:
        if total <= MAX_DIR_BYTES:
            break
        doomed.append(e)
        total -= sizes[e]
    for _, _, name in doomed:
        try:
            os.remove(os.path.join(d, name))
        except OSError:
            pass


def write(text: str, kind: str = "bash") -> str | None:
    """Write `text` to a fresh 0600 spill file and return its absolute path, or None
    if the write failed — spilling is best-effort, and a disk error must not turn a
    successful tool call into a broken result."""
    global _SWEPT
    d = session_dir()
    try:
        os.makedirs(d, exist_ok=True)
        if not _SWEPT:
            _SWEPT = True
            _sweep_stale(os.path.dirname(d))
        path = os.path.join(d, f"{kind}-{next(_IDS)}.log")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", errors="replace") as f:
            f.write(text)
        _prune(d)
        return os.path.abspath(path)
    except OSError:
        return None


def path_in(text: str) -> str | None:
    """The spill path an already-truncated body points at, or None.

    Verified two ways before it is trusted — the file must exist and must live under
    the spill root — because `text` can be tool output, and a result that merely
    quotes the phrase must never be handed back to the model as a live pointer."""
    if not isinstance(text, str):
        return None
    root = os.path.abspath(base_dir()) + os.sep
    for m in POINTER_RE.finditer(text):
        path = m.group(1)
        if os.path.abspath(path).startswith(root) and os.path.exists(path):
            return path
    return None
