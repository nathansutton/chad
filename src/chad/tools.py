"""chad's tool surface: bash, edit, write, write_todos, done.

The premise (measured, not aesthetic): the model already knows the unix toolbox
and the exact-match editor dialect from pretraining, so the harness exposes only
those and puts its knowledge into the RESULT channel (ambient.py) instead of
into more tools. A ranked-retrieval tool was built and measured against this
surface on a paired navigation benchmark: it left success unchanged at 6/6, did
not reach the answer any sooner (time-to-first-hit +1.1%), and cost 27.6% more
tool-result context. Discovery stays in bash.

Each tool has an OpenAI/Qwen-compatible JSON schema (exposed to the model via the
chat template's `tools` argument) and a Python implementation.
Implementations are deliberately conservative: writes/bash are real but the CLI
gates them behind a confirmation unless --yolo is set.
"""

import itertools
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any

from . import config, levers, seatbelt, syntaxgate
from .ignore import IGNORE_DIRS  # noqa: F401 — re-exported for agent.expand_mentions


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


# The only writable area in plan mode: plan files land here, nothing else may be
# touched. See the plan-mode gate in agent.run_turn.
PLANS_DIR = "plans"


def _under_plans(path: str) -> bool:
    """True if `path` resolves inside ./plans/ (the only writable area in plan mode)."""
    root = os.path.abspath(PLANS_DIR)
    p = os.path.abspath(path)
    return p == root or p.startswith(root + os.sep)


def _kill_group(p):
    """Kill the whole process group, not just the /bin/sh parent. `shell=True`
    spawns `/bin/sh -c <command>`; p.kill() SIGKILLs only that shell, leaving
    backgrounded/piped children (`cmd &`, `a | b`, a spawned server) alive — the
    exact long-running processes a timeout/interrupt exists to stop. start_new_session
    puts the shell in its own group so we can signal the whole tree."""
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        p.kill()  # group already gone, or no permission — fall back to the parent


# Environment variable names shaped like credentials, dropped from spawned shell
# children. Name-pattern only: values are never inspected (a value test would
# itself be a secret-handling liability), and the pattern is anchored at the end
# so AWS_SECRET_ACCESS_KEY, GITHUB_TOKEN, and DB_PASSWORD match while PATH,
# TOKENIZERS_PARALLELISM, and friends pass through. CHAD_NO_ENV_GUARD opts out
# for a session whose commands legitimately need a credential.
_ENV_SECRET_RE = re.compile(
    r"(?i)(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|API_?KEY|"
    r"ACCESS_KEY(_ID)?|SECRET_KEY|PRIVATE_KEY)$")


def _bash_env() -> dict | None:
    """The environment for a spawned bash child: a copy with credential-shaped
    names removed, or None (inherit the parent env untouched) under
    CHAD_NO_ENV_GUARD. A command that legitimately needs a credential sees a
    clear absence, not a corrupted value."""
    if config.flag("CHAD_NO_ENV_GUARD"):
        return None
    return {k: v for k, v in os.environ.items() if not _ENV_SECRET_RE.search(k)}


def tool_bash(command: str, timeout: int = 120, should_stop=None) -> str:
    argv = seatbelt.wrap_argv(command)
    try:
        # errors="replace": text mode decodes strictly by default, so binary bytes in the
        # output (hexdump, `cat` on an archive) killed the reader thread mid-communicate —
        # the command's whole output AND its exit code vanished into a "[no output]" that
        # read as a quiet success (and could spoof the verify gate).
        p = subprocess.Popen(argv if argv is not None else command,
                             shell=argv is None,
                             executable=config.shell_path() if argv is None else None,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                             start_new_session=True, env=_bash_env())
    except OSError as e:
        return f"[failed to launch: {e}]"
    # Drain output on a helper thread (so large output can't deadlock the pipe)
    # while we poll for an interrupt or timeout and kill the process if needed.
    box = {}
    t = threading.Thread(target=lambda: box.__setitem__("out", p.communicate()[0]),
                         daemon=True)
    t.start()
    deadline = time.time() + timeout
    while t.is_alive():
        if should_stop and should_stop():
            _kill_group(p); t.join(2)
            return _bash_killed("[interrupted by user", box.get("out"))
        if time.time() > deadline:
            _kill_group(p); t.join(2)
            return _bash_killed(f"[timed out after {timeout}s", box.get("out"))
        t.join(0.1)
    out = (box.get("out") or "").strip()
    if p.returncode is None:
        p.poll()  # reconcile: a reader-thread death skips communicate()'s internal wait
    if p.returncode not in (0, None):
        out = f"[exit {p.returncode}]\n{out}"
    out = _seatbelt_note(argv, out)
    return _bash_headtail(out) if out else "[no output]"


def _seatbelt_note(argv, out: str) -> str:
    """When a sandboxed command's output shows a write denial, say what happened
    and what to do — a bare 'Operation not permitted' reads as a broken tool, and
    a model that can't see the boundary will retry into it."""
    if argv is None or seatbelt.DENIAL_MARKER not in out:
        return out
    return out + seatbelt.DENIAL_NOTE


def _bash_killed(reason: str, partial: str | None) -> str:
    """Result for a bash command chad killed (timeout or interrupt). The output the
    process wrote before it died is already drained into `partial` by the join above —
    keeping it is the whole point: a build/train/download that was killed still printed
    how far it got (progress %, ETA, the last compile line, a hung-on-input prompt), and
    that partial output is exactly what tells the model whether to raise the timeout,
    background the command, or narrow the work. Discarding it (the old behavior) left the
    model to guess blind and re-run from zero — the dominant timeout-loss mode. Same
    head/tail clip as a normal result so a long partial can't blow up the prefill."""
    partial = (partial or "").strip()
    if not partial:
        return reason + "; no output before it was killed]"
    return (reason + f"; showing the {len(partial)} chars it printed before being killed — "
            "raise the timeout, background it (`cmd & … ; wait`), or narrow the work]\n"
            + _bash_headtail(partial))


# Bash output budget. A plain head-slice is exactly wrong for the thing bash is used
# for most — running tests/builds — because pytest/compilers put the actionable
# summary (`=== N failed ===`, the traceback tail) at the BOTTOM. On a noisy run a
# head-only cap shows 20k chars of passing dots and hides the failure, undermining the
# verify loop guardrails.py exists to enforce. So keep HEAD and TAIL (Claude Code does
# the same), biased toward the tail where the summary lives.
BASH_MAX_CHARS = 20000
BASH_HEAD_CHARS = 8000
BASH_TAIL_CHARS = 12000

# When output overflows the head/tail budget, the omitted middle used to be simply
# gone — the model's only way back to it (the first error of a cascade, one failing
# test among hundreds) was to re-run the command with a filter, paying the whole
# wall-clock again. So spill the FULL output to a session-scoped file and put the
# path in the truncation notice: a follow-up `grep marker <path>` costs seconds.
# Same philosophy as compaction._spill (dropped content goes to disk, not /dev/null).
# The dir is bounded two ways — chad has already had one unbounded-cache disk
# incident: each session keeps only its newest BASH_SPILL_KEEP files, and stale
# session dirs (orphaned by dead processes) are swept on the first spill.
BASH_SPILL_KEEP = 20
_SPILL_MAX_AGE_S = 7 * 24 * 3600
_SPILL_IDS = itertools.count(1)
_SPILL_SWEPT = False


def _spill_dir() -> str:
    """Per-process spill dir, resolved fresh each call (the eval harness repoints
    HOME per task). Never under cwd: a stray file would pollute the project's git
    status and fail the plan-mode untouched-tree check."""
    base = os.environ.get("CHAD_SPILL_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "chad", "spill")
    return os.path.join(base, str(os.getpid()))


def _sweep_stale_spills(base: str) -> None:
    """Delete sibling session dirs untouched for _SPILL_MAX_AGE_S. A spill is only
    useful to the session that wrote it; an old dir was orphaned by a dead process.
    Best-effort, runs once per process (on the first spill). Never raises."""
    try:
        deadline = time.time() - _SPILL_MAX_AGE_S
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


def _prune_spills(d: str) -> None:
    """Keep only the newest BASH_SPILL_KEEP spill files in `d` (by sequence number,
    which is monotonic within a process). Never raises."""
    def seq(name: str) -> int:
        try:
            return int(name[len("bash-"):-len(".log")])
        except ValueError:
            return -1
    try:
        names = sorted((n for n in os.listdir(d)
                        if n.startswith("bash-") and n.endswith(".log")), key=seq)
        for name in names[:-BASH_SPILL_KEEP]:
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass
    except OSError:
        pass


def _spill_bash(s: str) -> str | None:
    """Write the full untruncated output to a fresh 0600 spill file and return its
    absolute path, or None if the write failed — spilling is best-effort, and a
    disk error must not turn a successful command into a broken tool result."""
    global _SPILL_SWEPT
    d = _spill_dir()
    try:
        os.makedirs(d, exist_ok=True)
        if not _SPILL_SWEPT:
            _SPILL_SWEPT = True
            _sweep_stale_spills(os.path.dirname(d))
        path = os.path.join(d, f"bash-{next(_SPILL_IDS)}.log")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", errors="replace") as f:
            f.write(s)
        _prune_spills(d)
        return os.path.abspath(path)
    except OSError:
        return None


# Lines that name a specific failure — the rows a trimmed test/build output must not
# lose. Deliberately narrow: a marker glyph, a FAIL/ERROR word at a row's start, or the
# first line of a traceback. Anything looser matches the word "error" inside passing
# test NAMES ("✔ throws if …") and would spend the whole budget on rows that say nothing
# failed.
_FAILURE_LINE_RE = re.compile(
    r"^[ \t]*(?:[✘✗✖×]\s|\[?(?:FAIL|FAILED|ERROR|ERR)\b|not ok\b|E\s{2,}|"
    r"Traceback \(most recent call last\)|\S+:\d+:\d+:?\s+error\b|"
    r"\S+\(\d+,\d+\):\s+error\b).*$",
    re.M | re.I)
_TRIM_KEEP_MAX_LINES = 10
_TRIM_KEEP_MAX_CHARS = 700


def _kept_failures(middle: str) -> str:
    """Failure rows lifted out of the slice head/tail trimming is about to drop, or "".

    Head+tail is the right default (runners put the summary at the bottom), but it
    assumes the middle is filler. For a test runner it isn't: ava prints each failing
    test's NAME and assertion in the body and only the count at the end, so a trimmed
    run can say `2 tests failed` while the two names it failed on are exactly what got
    cut. Measured on a real run — the model saw the counts, never the names."""
    hits = []
    seen = set()
    for m in _FAILURE_LINE_RE.finditer(middle):
        line = m.group(0).strip()[:200]
        if line and line not in seen:
            seen.add(line)
            hits.append(line)
        if len(hits) >= _TRIM_KEEP_MAX_LINES:
            break
    if not hits:
        return ""
    out = "\n".join(hits)
    return out[:_TRIM_KEEP_MAX_CHARS - 1] + "…" if len(out) > _TRIM_KEEP_MAX_CHARS else out


# Per-line cap for bash output. head+tail budgets the TOTAL, which silently assumes
# output is made of lines; a source map, a minified bundle, or a one-line JSON blob is a
# single line of tens of thousands of chars, and the budget then hands back 20k chars
# sliced out of the middle of base64. Measured: one `rg --no-ignore` that reached a
# gitignored `distribution/` returned a 92k-char .js.map line, of which 20k was prefilled
# — ~5k tokens and a 147s stall for zero information. `tool_grep` has capped its rows at
# GREP_LINE_CHARS since it shipped; this is the same guarantee for the route that has no
# grep tool. Slightly looser than grep's 500 because bash carries wrapped compiler errors
# and stack frames that are legitimately long.
BASH_LINE_CHARS = 600
# Clipping this much means the transcript no longer holds the output, so the full text
# goes to a spill file the model can grep — never silently destroyed.
BASH_CLIP_SPILL_AT = BASH_MAX_CHARS // 4


def _clip_long_lines(s: str) -> tuple[str, int]:
    """(text with over-long lines capped, chars removed). Byte-identical and 0 when
    every line already fits, so the common case pays only the scan."""
    if not levers.enabled("bash_line_clip"):
        return s, 0
    removed = 0
    out = []
    for line in s.split("\n"):
        if len(line) > BASH_LINE_CHARS:
            removed += len(line) - BASH_LINE_CHARS
            line = line[:BASH_LINE_CHARS] + "…[line clipped]"
        out.append(line)
    return ("\n".join(out), removed) if removed else (s, 0)


def _bash_headtail(s: str, spill: bool = True) -> str:
    # Clip pathological lines FIRST: the head/tail budget below assumes the output is
    # made of lines, and one 92k-char source-map line defeats it completely. Whatever
    # spills is the ORIGINAL text, never the clipped view — a spill file the model is
    # told to grep must hold what the command actually printed.
    clipped, removed = _clip_long_lines(s)
    clip_note, path = "", None
    if removed:
        levers.fired("bash_line_clip", removed=removed)
        if spill and removed >= BASH_CLIP_SPILL_AT:
            path = _spill_bash(s)
        where = (f"; FULL output saved to {path}; grep/sed that file for the rest"
                 if path else "")
        clip_note = (f"\n[… {removed} chars clipped from over-long lines "
                     f"(each capped at {BASH_LINE_CHARS} chars){where} …]")
    s = clipped
    if len(s) + len(clip_note) <= BASH_MAX_CHARS:
        return s + clip_note
    omitted = len(s) - BASH_HEAD_CHARS - BASH_TAIL_CHARS
    if path is None and spill:
        path = _spill_bash(s)
    if path:
        where = (f"FULL output saved to {path}; grep/sed that file instead of "
                 f"re-running the command")
    else:
        where = "output truncated"
    kept = ""
    if levers.enabled("bash_trim_keep_failures"):
        kept = _kept_failures(s[BASH_HEAD_CHARS:len(s) - BASH_TAIL_CHARS])
    if kept:
        levers.fired("bash_trim_keep_failures", lines=kept.count("\n") + 1)
        notice = (f"\n[… {omitted} chars omitted — {where}. The failure rows from the "
                  f"omitted middle, kept verbatim …]\n{kept}\n"
                  f"[… end of kept rows; the TAIL below is usually the summary …]\n")
    else:
        notice = (f"\n[… {omitted} chars omitted — {where}. The TAIL below is usually "
                  f"the failure summary …]\n")
    return s[:BASH_HEAD_CHARS] + notice + s[-BASH_TAIL_CHARS:] + clip_note


def tool_write(path: str, content: str) -> str:
    before = None
    if os.path.exists(path):
        try:
            with open(path, errors="replace") as f:
                before = f.read()
        except OSError:
            pass
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    result = f"[wrote {len(content)} bytes to {_rel(path)}]"
    warn = syntaxgate.check_syntax(path, before)
    if warn:
        result += warn
    return result


# Edit robustness. Dogfooding logs showed ~1 in 6 `edit` calls failed to apply —
# dominated by two MECHANICAL near-misses, not bad intent: (1) the model emits literal
# "\n"/"\t" in `old` instead of real control chars (so a multiline `old` never matches),
# and (2) indentation / trailing-whitespace drift between what it quoted and the file.
# Both are recoverable WITHOUT risking a wrong edit, because each recovery still requires
# a UNIQUE target — we never replace on an ambiguous or fuzzy-multiple match. A miss that
# can't be resolved now returns the closest line in the file so the model can self-correct
# instead of looping on the identical bad call.

def _unescape_ws(s: str) -> str:
    """Interpret the literal backslash escapes a weak model emits (\\n \\t \\r) as the
    real control chars. Targeted, not a blanket unicode_escape (which would mangle real
    backslashes and unicode in code)."""
    return s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _line_offsets(data: str):
    offs, pos = [], 0
    for ln in data.split("\n"):
        offs.append(pos)
        pos += len(ln) + 1  # +1 for the stripped '\n'
    return offs


def _ws_flexible_spans(data: str, old: str):
    """Char spans (start, end) where `old` matches a run of lines in `data` ignoring
    each line's leading/trailing whitespace. Skips all-blank patterns (too
    ambiguous)."""
    fold = lambda l: l.strip()  # noqa: E731
    norm = [fold(l) for l in old.strip("\n").split("\n")]
    if not any(norm):
        return []
    dlines = data.split("\n")
    offs = _line_offsets(data)
    n = len(norm)
    spans = []
    for i in range(len(dlines) - n + 1):
        if [fold(dlines[i + j]) for j in range(n)] == norm:
            spans.append((offs[i], offs[i + n - 1] + len(dlines[i + n - 1])))
    return spans


def _reindent(new: str, target_indent: str, span_text: str | None = None,
              indent_unit: str | None = None) -> str:
    """Shift `new` so its first non-blank line carries target_indent, preserving the
    relative indentation of the rest (so a recovered block lands at the file's indent).

    When `span_text` (the file text being replaced) is given, prefer the FILE's own
    indentation over the model's: a same-line-count replacement takes each span
    line's indent positionally, and any line whose stripped content matches a span
    line takes that line's indent. On the whitespace-flexible recovery path the
    model's relative indents are the least trustworthy part of the edit — the
    demonstrated failure: a semantically correct one-line fix landed
    with the model's broken 10-space indent, shipped an IndentationError, and the
    resulting file was unrepairable through this same path.

    Inserted lines — lines in `new` with no counterpart in the span — used to take the
    FIRST span line's indent (`target_indent`) via first-line math, which is irrelevant
    to a line inserted deep inside the block (ky-timeoutMessage, session
    dbf9dee0/20260713: `timeoutMessage: true,` inserted between two `\\t`-indented
    siblings landed at the wrong tab depth, then cost ~35 tool calls of tab surgery).
    Now each run of unresolved lines is NEIGHBOR-ANCHORED: it takes the file indent of
    the nearest resolved line before it (else after it), one `indent_unit` deeper when
    that neighbor opens a block (`{ ( [` or a trailing `:`). The model's relative depth
    between unresolved lines is deliberately ignored — in that session the model believed
    a sibling belonged one level deeper, so honoring relative depth reproduces the bug;
    a multi-line nested insert may therefore land under-indented, but the recovery-path
    echo makes that visible and a whitespace-only follow-up applies verbatim."""
    def _ind(s: str) -> str:
        return s[: len(s) - len(s.lstrip())]
    lines = new.split("\n")
    span_lines = span_text.split("\n") if span_text is not None else []
    positional = len(span_lines) == len(lines) and bool(span_lines)
    strip_map: dict[str, str] = {}
    for sl in span_lines:
        if sl.strip():
            strip_map.setdefault(sl.strip(), _ind(sl))

    # Pass 1: resolve each non-blank line's indent from the FILE where a counterpart
    # exists (positional map, else stripped-content map). `resolved[i]` is None for a
    # blank line or an inserted line with no file counterpart.
    resolved: list[str | None] = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            resolved.append(None)
        elif positional and span_lines[i].strip():
            resolved.append(_ind(span_lines[i]))
        elif ln.strip() in strip_map:
            resolved.append(strip_map[ln.strip()])
        else:
            resolved.append(None)

    # Pass 2: emit. Resolved lines take their file indent; unresolved lines are
    # neighbor-anchored (above). When nothing at all resolved (no span / nothing
    # matched) fall back to the legacy first-line math, byte-for-byte unchanged.
    any_resolved = any(r is not None for r in resolved)
    first = next((l for l in lines if l.strip()), "")
    src = _ind(first)
    out = []
    for i, ln in enumerate(lines):
        r = resolved[i]
        if not ln.strip():
            out.append("")
        elif r is not None:
            out.append(r + ln.strip())
        elif not any_resolved:
            if ln.startswith(src):
                out.append(target_indent + ln[len(src):])
            else:
                out.append(target_indent + ln.lstrip())
        else:
            anchor = _nearest_resolved(lines, resolved, i)
            base = anchor[0]
            if anchor[1] and indent_unit and anchor[2].rstrip().endswith(("{", "(", "[", ":")):
                base = base + indent_unit
            out.append(base + ln.strip())
    return "\n".join(out)


def _nearest_resolved(lines: list[str], resolved: list[str | None],
                      i: int) -> tuple[str, bool, str]:
    """Indent of the nearest resolved non-blank line to `i`, preferring the one BEFORE
    it (so an insert sits with the sibling it follows). Returns (indent, is_before,
    anchor_stripped); the opener bump only applies to a before-anchor. Never called
    unless some line resolved, so the final `("", ...)` is unreachable in practice."""
    for j in range(i - 1, -1, -1):
        if resolved[j] is not None:
            return resolved[j], True, lines[j].strip()  # type: ignore[return-value]
    for j in range(i + 1, len(lines)):
        if resolved[j] is not None:
            return resolved[j], False, lines[j].strip()  # type: ignore[return-value]
    return "", False, ""


def _closest_hint(data: str, old: str) -> str:
    import difflib
    target = old.strip("\n").split("\n")[0].strip()
    if not target:
        return ""
    best = difflib.get_close_matches(
        target, [l.strip() for l in data.split("\n") if l.strip()], n=1, cutoff=0.6)
    if not best:
        return ""
    return f" Closest line in the file is {best[0]!r} — copy it exactly."


def _show_ws(line: str) -> str:
    """Render a file line with its leading whitespace made visible (· = space, → = tab)
    so the model copies the exact indentation instead of re-guessing the column count —
    the space-miscount that drives the no-op edit loop when it hand-patches indentation."""
    stripped = line.lstrip(" \t")
    indent = line[: len(line) - len(stripped)]
    return indent.replace("\t", "→").replace(" ", "·") + stripped


def _locate_block(data: str, old: str):
    """(index of the file line `old` was probably aiming at, the file's lines), or
    (None, lines). Exact stripped match on `old`'s first non-blank line, then a fuzzy
    one. Shared by every failed-edit hint so they all describe the SAME region — a
    diagnosis that points somewhere the indentation echo doesn't is worse than none."""
    dlines = data.split("\n")
    key = next((l.strip() for l in old.strip("\n").split("\n") if l.strip()), "")
    if not key:
        return None, dlines
    idx = next((i for i, l in enumerate(dlines) if l.strip() == key), None)
    if idx is None:
        import difflib
        near = difflib.get_close_matches(
            key, [l.strip() for l in dlines if l.strip()], n=1, cutoff=0.6)
        if not near:
            return None, dlines
        idx = next((i for i, l in enumerate(dlines) if l.strip() == near[0]), None)
    return idx, dlines


def _already_applied_hint(data: str, old: str, new: str) -> str:
    """An observation when every line `new` would ADD is already in the file: this site
    may have been edited already. Measured: on a nine-site threading edit the model lost
    track, re-sent one site's original text, and read "closest line … copy it exactly
    (mind indentation)" as an indentation problem — so it went hunting for tabs instead
    of noticing the change was in. Stated as an observation, never a verdict: the added
    line can also legitimately exist at a DIFFERENT site of the same repeated edit."""
    added = [l.strip() for l in new.strip("\n").split("\n")
             if l.strip() and l.strip() not in {o.strip() for o in old.split("\n")}]
    if not added:
        return ""
    dstripped = [l.strip() for l in data.split("\n")]
    where = []
    for line in added:
        if line not in dstripped:
            return ""
        where.append(dstripped.index(line) + 1)
    return (f"\n[note: every line your `new` would add is ALREADY in this file "
            f"(first at line {where[0]}) — check whether this site is already edited, "
            f"or whether you are looking at a different site of the same change.]")


def _first_diff_hint(data: str, old: str) -> str:
    """Where `old` first stops matching the file region it was aiming at: line, column,
    and both texts with whitespace made visible. The generic "copy it exactly (mind
    indentation)" names one hypothesis, and when it is the wrong one the model spends
    its next calls on it — measured twice in one run: a stray leading quote read as a
    tab problem (two identical retries, then `cat -A`, then `sed -n l`), and a
    hallucinated second line read the same way."""
    idx, dlines = _locate_block(data, old)
    if idx is None:
        return ""
    olines = old.strip("\n").split("\n")
    for n, oline in enumerate(olines):
        fline = dlines[idx + n] if idx + n < len(dlines) else ""
        if oline == fline:
            continue
        col = next((c for c in range(min(len(oline), len(fline)) + 1)
                    if c == len(oline) or c == len(fline) or oline[c] != fline[c]), 0)
        which = "your first line" if n == 0 else f"line {n + 1} of your `old`"
        return (f"\n[{which} differs from the file at column {col + 1}:\n"
                f"  you sent: {_show_ws(oline)[:120]}\n"
                f"  the file: {_show_ws(fline)[:120]}\n]")
    return ""


def _edit_miss_hints(data: str, old: str, new: str) -> str:
    """The two diagnoses a bare "old string not found" leaves the model to derive."""
    if not levers.enabled("edit_miss_diagnose"):
        return ""
    hint = _already_applied_hint(data, old, new) or _first_diff_hint(data, old)
    if hint:
        levers.fired("edit_miss_diagnose",
                     kind="already_applied" if hint.startswith("\n[note:") else "first_diff")
    return hint


def _indent_hint(data: str, old: str) -> str:
    """Echo the run of file lines that `old` was trying to match, with leading whitespace
    made visible, so a failed/no-op edit hands back the exact current indentation to copy.
    Empty when no plausible location is found."""
    olines = old.strip("\n").split("\n")
    idx, dlines = _locate_block(data, old)
    if idx is None:
        return ""
    region = dlines[idx: idx + max(1, len(olines))]
    shown = "\n".join(_show_ws(l) for l in region)
    if len(shown) > 800:
        shown = shown[:800] + "…"
    return ("\n[current lines in the file (· = one space, → = one tab) — copy this "
            f"indentation exactly:\n{shown}\n]")


def _landed_hint(block: str) -> str:
    """Echo the replacement block that just landed with leading whitespace made visible
    (· = space, → = tab), so a recovery-path edit SHOWS the model the depth it chose
    instead of it verifying via extra read/sed/od round-trips (ky-timeoutMessage, session
    dbf9dee0/20260713: ~10 blind whitespace-check rounds after a recovered insert). Same
    legend wording as `_indent_hint`, capped at 800 chars, so the model sees one format."""
    shown = "\n".join(_show_ws(l) for l in block.split("\n"))
    if len(shown) > 800:
        shown = shown[:800] + "…"
    return f"\n[landed lines (· = one space, → = one tab):\n{shown}\n]"


def _apply_edit(path: str, before: str, after: str, note: str) -> str:
    if after == before:
        return "[no-op edit: the replacement leaves the file unchanged]"
    with open(path, "w") as f:
        f.write(after)
    result = f"[edited {_rel(path)}{note}]"
    warn = syntaxgate.check_syntax(path, before)
    if warn:
        result += warn
    return result


def tool_edit(path: str, old: str, new: str) -> str:
    if not os.path.exists(path):
        return f"[no such file: {path}]"
    with open(path) as f:
        data = f.read()
    if old == new:
        return ("[no-op edit: old and new are identical; change the content or stop]"
                + _indent_hint(data, old))

    # (1) exact match — the common, fast path.
    n = data.count(old)
    if n == 1:
        return _apply_edit(path, data, data.replace(old, new, 1), "")
    if n > 1:
        return f"[old string appears {n} times; make it unique by including more surrounding lines]"

    # (2) escape-normalized: literal \n/\t in `old` (and `new` when it has no real newline).
    uold = _unescape_ws(old)
    unew = _unescape_ws(new) if ("\n" not in new and "\\n" in new) else new
    # STOP condition: whether a literal `\n` in `new` is an escape the
    # model meant as a newline, or a genuine backslash-n it wants written verbatim, is
    # ambiguous — and `unew` is only ever used on a recovery path where `old` itself
    # needed the same unescape, so we keep the historical transform but DISCLOSE it in the
    # result so the model can correct a mis-transformed literal instead of it happening
    # silently. `note_new` is the disclosure fragment, empty when `new` was left as-is.
    note_new = (" [note: \\n in replacement interpreted as newline; re-edit with a real "
                "newline if you meant a literal backslash-n]") if unew != new else ""
    if uold != old:
        c = data.count(uold)
        if c == 1:
            return _apply_edit(path, data, data.replace(uold, unew, 1),
                               " (recovered: interpreted \\n/\\t escapes in `old`)" + note_new)
        if c > 1:
            return f"[old string appears {c} times; make it unique by including more surrounding lines]"

    # (3) whitespace-flexible: indentation / trailing-space drift, still requiring uniqueness.
    probe = uold if uold != old else old
    spans = _ws_flexible_spans(data, probe)
    how = "indentation/whitespace"
    if len(spans) == 1:
        s, e = spans[0]
        head = data[s:e].split("\n")[0]
        indent = head[: len(head) - len(head.lstrip())]
        used_unew = uold != old  # this path only unescapes `new` when `old` was unescaped
        raw = (unew if used_unew else new).strip("\n")
        repl = _reindent(raw, indent, data[s:e], indent_unit=_indent_unit(data))
        if data[:s] + repl + data[e:] == data and raw != data[s:e]:
            # Reindenting reproduced the file byte-for-byte, yet the model's `new`
            # differs from the span — the edit IS a whitespace change (an
            # indentation fix). Normalizing it away made a broken indent literally
            # unrepairable through this tool (measured: every fix attempt returned
            # "[no-op edit]" and the model fell back to blind sed). Trust
            # the model's whitespace verbatim.
            res = _apply_edit(path, data, data[:s] + raw + data[e:],
                              " (applied verbatim: whitespace-only change)"
                              + (note_new if used_unew else ""))
            landed = raw
        else:
            res = _apply_edit(path, data, data[:s] + repl + data[e:],
                              f" (recovered: matched ignoring {how})"
                              + (note_new if used_unew else ""))
            landed = repl
        # Echo the landed region with visible whitespace (ky-timeoutMessage, session
        # dbf9dee0/20260713): recovery reindents, so the model must SEE the depth that
        # landed rather than re-derive it via read/sed/od. Only on a result that landed —
        # `_apply_edit` can return a syntaxgate rejection or drift warning instead.
        return res + _landed_hint(landed) if res.startswith("[edited") else res
    if len(spans) > 1:
        return (f"[old string matches {len(spans)} places ignoring {how}; include "
                f"more surrounding lines to make it unique]")

    return (f"[old string not found; no change made.{_closest_hint(data, old)}]"
            + _indent_hint(data, old) + _edit_miss_hints(data, old, new))


def _indent_unit(data: str) -> str:
    """The file's own one-level indent string: a tab when its indented lines lead with
    tabs, else the smallest positive run of leading spaces seen (clamped to 2/4/8, default
    4). Lets the indentation recoveries rebuild whitespace in the FILE's unit instead of
    assuming spaces — a tab-indented file is what turned '_fit_indent' into a tab/space
    mixer before this."""
    tab = space = 0
    widths = []
    for ln in data.split("\n"):
        if not ln[:1].isspace():
            continue
        lead = ln[: len(ln) - len(ln.lstrip())]
        if lead[:1] == "\t":
            tab += 1
        else:
            space += 1
            widths.append(len(lead) - len(lead.lstrip(" ")))
    if tab > space:
        return "\t"
    pos = [w for w in widths if w > 0]
    step = min(pos) if pos else 4
    return " " * (step if step in (2, 4, 8) else 4)


# Planning tool (deepagents' write_todos): a scaffold that keeps the model on track
# across multi-step tasks. Stateless-ish — the model re-sends the whole list each call.
_TODOS = []


def tool_write_todos(todos) -> str:
    global _TODOS
    if not isinstance(todos, list):
        return "[todos must be a list of {content, status} objects]"
    _TODOS = todos
    marks = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
    lines = [f"  {marks.get(t.get('status', 'pending'), '[ ]')} {t.get('content', '')}"
             for t in todos]
    return "Plan updated:\n" + "\n".join(lines)


# Each entry takes (args, should_stop); long-running tools honor should_stop so a
# ctrl-c interrupt can abort them mid-flight.
DISPATCH = {
    "write_todos": lambda a, ss=None: tool_write_todos(a["todos"]),
    "bash": lambda a, ss=None: tool_bash(a["command"], a.get("timeout", 120), should_stop=ss),
    "write": lambda a, ss=None: tool_write(a["path"], a["content"]),
    "edit": lambda a, ss=None: tool_edit(a["path"], a["old"], a["new"]),
}


# Tools that mutate state -> CLI asks for confirmation unless --yolo.
MUTATING = {"bash", "write", "edit"}

# Terminal tools end the turn cleanly (forge's terminal_tool idea). Small models
# instinctively try to "stop"/"finish"; giving them a real tool avoids hallucinated
# unknown-tool churn at the end of a task.
TERMINAL = {"done", "finish", "stop"}

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_todos",
            "description": "Record or update your step-by-step plan for a multi-step task. "
                           "Call this first for any task with 2+ steps, and again to update "
                           "statuses as you progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string",
                                           "enum": ["pending", "in_progress", "completed"]},
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call this when the task is fully complete and verified, to end "
                           "your turn. Provide a one-line summary of what you did.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the working directory and return combined stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write (create or overwrite) a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace a unique substring in a file with new text. Requires "
                           "an EXACT match of `old` including indentation — copy the "
                           "current text from what you just read, and include enough "
                           "surrounding lines to make `old` unique in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to replace (must be unique)."},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]


def active_schemas():
    """The tool schemas to expose to the model right now: SCHEMAS plus any connected MCP
    server's tools.

    Agent Skills add nothing here. They are invoked by the user typing `/name`, which
    submits the skill body as a user turn — so there is no skill-selection tool for the
    model, and no per-skill cost in the prompt (see skills.slash_commands)."""
    schemas = SCHEMAS
    # Tools from connected MCP servers (mcp__<server>__<tool>), if any are configured.
    # Empty list when none, so a server-less project never sees an extra tool.
    mcp_schemas = _mcp().schemas()
    if mcp_schemas:
        schemas = schemas + mcp_schemas
    return schemas


def _mcp():
    """Lazy import of the mcp module (avoids a circular import at module load)."""
    from . import mcp
    return mcp


def dispatch_for(name):
    """Return the callable (args, should_stop)->str that handles a tool call, checking
    chad's builtin DISPATCH first and then connected MCP servers. None if the name is
    not a known tool (the agent then runs the unknown-tool repair path)."""
    fn = DISPATCH.get(name)
    if fn is not None:
        return fn
    if _mcp().is_mcp_tool(name) and _mcp().has_tool(name):
        return lambda a, ss=None: _mcp().call(name, a)
    return None


def is_mutating(name) -> bool:
    """Whether a tool call needs the confirmation gate: a builtin mutator, or an MCP
    tool the server didn't mark read-only (see mcp._is_mutating)."""
    return name in MUTATING or _mcp().is_mutating(name)
