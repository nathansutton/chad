"""Claude-Code-style tools for the agent loop.

Each tool has an OpenAI/Qwen-compatible JSON schema (exposed to the model via the
chat template's `tools` argument) and a Python implementation. Implementations are
deliberately conservative: reads are unrestricted, writes/bash are real but the CLI
gates them behind a confirmation unless --yolo is set.
"""

import fnmatch
import glob as _glob
import io
import os
import re
import signal
import subprocess
import threading
import time
from typing import Any

from . import config, levers, repomap, symbols, syntaxgate
from .ignore import IGNORE_DIRS, slash_wrapped

# Directories never worth walking: huge, generated, or VCS internals. The canonical set
# of bare names lives in `ignore.py` (the single source of truth); `IGNORE_DIRS` is
# re-exported here because agent.expand_mentions imports it to filter @dir listings.
# `_SKIP_DIRS` is the slash-wrapped form for path-substring tests.
_SKIP_DIRS = slash_wrapped(IGNORE_DIRS)


def _skip(path: str) -> bool:
    p = "/" + path.replace(os.sep, "/")
    return any(d in p for d in _SKIP_DIRS)


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


def tool_bash(command: str, timeout: int = 120, should_stop=None) -> str:
    try:
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
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
            return "[interrupted by user]"
        if time.time() > deadline:
            _kill_group(p); t.join(2)
            return f"[timed out after {timeout}s]"
        t.join(0.1)
    out = (box.get("out") or "").strip()
    if p.returncode not in (0, None):
        out = f"[exit {p.returncode}]\n{out}"
    return _bash_headtail(out) if out else "[no output]"


# Bash output budget. A plain head-slice is exactly wrong for the thing bash is used
# for most — running tests/builds — because pytest/compilers put the actionable
# summary (`=== N failed ===`, the traceback tail) at the BOTTOM. On a noisy run a
# head-only cap shows 20k chars of passing dots and hides the failure, undermining the
# verify loop guardrails.py exists to enforce. So keep HEAD and TAIL (Claude Code does
# the same), biased toward the tail where the summary lives.
BASH_MAX_CHARS = 20000
BASH_HEAD_CHARS = 8000
BASH_TAIL_CHARS = 12000


def _bash_headtail(s: str) -> str:
    if len(s) <= BASH_MAX_CHARS:
        return s
    omitted = len(s) - BASH_HEAD_CHARS - BASH_TAIL_CHARS
    return (s[:BASH_HEAD_CHARS]
            + f"\n[… {omitted} chars omitted — output truncated; the TAIL below is "
              f"usually the failure summary …]\n"
            + s[-BASH_TAIL_CHARS:])


# Local-model read budget. Every token a read returns must be PREFILLED into the
# KV cache, and on Ornith (~50 tok/s, ~350 tok/s prefill) a whole-file dump of a
# 1377-line test file = ~21k tokens = a ~60s stall AND a multi-GB transient-memory
# spike during the prefill. Cloud Claude Code can afford 2000-line reads because of
# prompt caching; locally we default small and page, nudging toward the symbolic
# tools (find_symbol/view_symbol/grep) for big files. A hard char cap also clips
# pathological long lines (minified/data files).
READ_DEFAULT_LIMIT = 400
READ_MAX_LIMIT = 800
# Hard cap on the chars any single read appends to the transcript. This is the one
# uncapped path for NON-code files: skeleton mode (below) shrinks big parseable code,
# but a large README/markdown/doc has no skeleton and would otherwise dump its whole
# body. Dogfooding showed a ~24k-char read = ~8k tokens = a ~25s prefill stall before
# the next turn speaks. Cap at ~10k chars (≈3k tokens): still a generous view, ~10s
# worst case; the note tells the model to page (offset=) or grep for the rest.
READ_MAX_CHARS = 10000
# Above this many lines, a default (un-paged) read of a parseable code file returns
# its SKELETON (signatures) instead of the body. The eval data showed the model
# defaults to `read` even when view_symbol is cheaper, so the harness — not the
# model — caps the cost: a skeleton is ~10x smaller and the bodies are one
# view_symbol / offset-read away. Set above the size of normal source files so
# small reads are untouched; big files can't blow up prefill.
READ_SKELETON_LINES = 250


def tool_read(path: str, offset: int = 0, limit: int = READ_DEFAULT_LIMIT) -> str:
    # Small models often emit a workspace-relative path with a stray leading slash
    # ("/construct.py"). If the absolute path doesn't exist but the relative one does,
    # fall back to it rather than a misleading "[no such file]".
    if not os.path.exists(path) and path.startswith("/") and os.path.exists(path.lstrip("/")):
        path = path.lstrip("/")
    if not os.path.exists(path):
        return f"[no such file: {path}]"
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except (IsADirectoryError, OSError) as e:
        return f"[cannot read {path}: {e}]"
    total = len(lines)

    # Skeleton mode: a plain "read this big file" returns structure, not the whole
    # body — keeps prefill bounded without relying on the model to pick view_symbol.
    # Only for default reads (no explicit window) of large, parseable code files.
    if offset == 0 and limit == READ_DEFAULT_LIMIT and total > READ_SKELETON_LINES:
        skel = repomap.service().overview(path)
        if skel and not skel.startswith("["):  # had real functions/classes
            return (f"[NOT an error — this is the STRUCTURE of {_rel(path)} "
                    f"({total} lines): every function/class signature with line "
                    f"numbers, bodies omitted to keep context small. The read worked. "
                    f"To see a body: view_symbol(name), or read(path, offset=N, "
                    f"limit=M) with the line numbers below.]\n{skel}")

    # Paging a big code file with offset= defeats the skeleton guard above and racks up
    # prefill page by page. Nudge toward symbol-targeted reads (the bigfile tasks show
    # this is the expensive losing move on a non-trimmable cache).
    lead = ""
    if offset > 0 and total > READ_SKELETON_LINES and repomap.service().lang_for(path):
        lead = (f"[paging a {total}-line code file — view_symbol(name) returns just the one "
                f"function (~10x cheaper than reading pages); overview({_rel(path)}) lists "
                f"the symbols.]\n")

    limit = max(1, min(limit, READ_MAX_LIMIT))
    chunk = lines[offset : offset + limit]
    width = len(str(offset + len(chunk)))
    body = "".join(f"{i+offset+1:>{width}}  {ln}" for i, ln in enumerate(chunk))
    note = ""
    if len(body) > READ_MAX_CHARS:  # clip long-line blobs before they bloat context
        body = body[:READ_MAX_CHARS]
        note = (f"\n[…clipped at {READ_MAX_CHARS} chars. This file is dense — use grep "
                f"or find_symbol/view_symbol to target what you need.]")
    shown_end = offset + len(chunk)
    if shown_end < total:  # more file remains past the window we returned
        note += (f"\n[showed lines {offset+1}-{shown_end} of {total}. To continue, read "
                 f"with offset={shown_end}; or use grep/find_symbol to jump to what you "
                 f"need instead of reading the whole file.]")
    return (lead + body + note) if body else "[empty]"


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
    return result + warn if warn else result


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
    each line's leading/trailing whitespace. Skips all-blank patterns (too ambiguous)."""
    norm = [l.strip() for l in old.strip("\n").split("\n")]
    if not any(norm):
        return []
    dlines = data.split("\n")
    offs = _line_offsets(data)
    n = len(norm)
    spans = []
    for i in range(len(dlines) - n + 1):
        if [dlines[i + j].strip() for j in range(n)] == norm:
            spans.append((offs[i], offs[i + n - 1] + len(dlines[i + n - 1])))
    return spans


def _reindent(new: str, target_indent: str, span_text: str | None = None) -> str:
    """Shift `new` so its first non-blank line carries target_indent, preserving the
    relative indentation of the rest (so a recovered block lands at the file's indent).

    When `span_text` (the file text being replaced) is given, prefer the FILE's own
    indentation over the model's: a same-line-count replacement takes each span
    line's indent positionally, and any line whose stripped content matches a span
    line takes that line's indent. On the whitespace-flexible recovery path the
    model's relative indents are the least trustworthy part of the edit — the
    demonstrated failure (sphinx-7440): a semantically correct one-line fix landed
    with the model's broken 10-space indent, shipped an IndentationError, and the
    resulting file was unrepairable through this same path."""
    def _ind(s: str) -> str:
        return s[: len(s) - len(s.lstrip())]
    lines = new.split("\n")
    span_lines = span_text.split("\n") if span_text is not None else []
    positional = len(span_lines) == len(lines) and bool(span_lines)
    strip_map: dict[str, str] = {}
    for sl in span_lines:
        if sl.strip():
            strip_map.setdefault(sl.strip(), _ind(sl))
    first = next((l for l in lines if l.strip()), "")
    src = _ind(first)
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            out.append("")
        elif positional and span_lines[i].strip():
            out.append(_ind(span_lines[i]) + ln.strip())
        elif ln.strip() in strip_map:
            out.append(strip_map[ln.strip()] + ln.strip())
        elif ln.startswith(src):
            out.append(target_indent + ln[len(src):])
        else:
            out.append(target_indent + ln.lstrip())
    return "\n".join(out)


def _closest_hint(data: str, old: str) -> str:
    import difflib
    target = old.strip("\n").split("\n")[0].strip()
    if not target:
        return ""
    best = difflib.get_close_matches(
        target, [l.strip() for l in data.split("\n") if l.strip()], n=1, cutoff=0.6)
    return (f" Closest line in the file is {best[0]!r} — copy it exactly (mind "
            f"indentation), or use replace_lines(path, start, end, new) with the line "
            f"numbers from read (it fits indentation for you), or replace_symbol to "
            f"rewrite the whole function.") if best else ""


def _show_ws(line: str) -> str:
    """Render a file line with its leading whitespace made visible (· = space, → = tab)
    so the model copies the exact indentation instead of re-guessing the column count —
    the space-miscount that drives the no-op edit loop when it hand-patches indentation."""
    stripped = line.lstrip(" \t")
    indent = line[: len(line) - len(stripped)]
    return indent.replace("\t", "→").replace(" ", "·") + stripped


def _indent_hint(data: str, old: str) -> str:
    """Echo the run of file lines that `old` was trying to match, with leading whitespace
    made visible, so a failed/no-op edit hands back the exact current indentation to copy.
    Empty when no plausible location is found."""
    olines = old.strip("\n").split("\n")
    key = next((l.strip() for l in olines if l.strip()), "")
    if not key:
        return ""
    dlines = data.split("\n")
    idx = next((i for i, l in enumerate(dlines) if l.strip() == key), None)
    if idx is None:
        import difflib
        near = difflib.get_close_matches(
            key, [l.strip() for l in dlines if l.strip()], n=1, cutoff=0.6)
        if not near:
            return ""
        idx = next((i for i, l in enumerate(dlines) if l.strip() == near[0]), None)
        if idx is None:
            return ""
    region = dlines[idx: idx + max(1, len(olines))]
    shown = "\n".join(_show_ws(l) for l in region)
    if len(shown) > 800:
        shown = shown[:800] + "…"
    return ("\n[current lines in the file (· = one space, → = one tab) — copy this "
            f"indentation exactly:\n{shown}\n]")


def _apply_edit(path: str, before: str, after: str, note: str) -> str:
    if after == before:
        return "[no-op edit: the replacement leaves the file unchanged]"
    # Prong 1 (plan 067): never LAND an edit that newly breaks Python indentation —
    # revert and make the model re-send. A landed indent/tab break is the precondition
    # for the whitespace-surgery loop it can't win. Scoped to IndentationError; a clean
    # file only (an already-broken file stays editable so real fixes aren't stranded).
    reject = syntaxgate.indent_reject(path, before, after)
    if reject:
        return reject
    with open(path, "w") as f:
        f.write(after)
    result = f"[edited {_rel(path)}{note}]"
    warn = syntaxgate.check_syntax(path, before)
    return result + warn if warn else result


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
    # STOP condition (plan 044 item 4): whether a literal `\n` in `new` is an escape the
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
    if len(spans) == 1:
        s, e = spans[0]
        head = data[s:e].split("\n")[0]
        indent = head[: len(head) - len(head.lstrip())]
        used_unew = uold != old  # this path only unescapes `new` when `old` was unescaped
        raw = (unew if used_unew else new).strip("\n")
        repl = _reindent(raw, indent, data[s:e])
        if data[:s] + repl + data[e:] == data and raw != data[s:e]:
            # Reindenting reproduced the file byte-for-byte, yet the model's `new`
            # differs from the span — the edit IS a whitespace change (an
            # indentation fix). Normalizing it away made a broken indent literally
            # unrepairable through this tool (sphinx-7440: every fix attempt
            # returned "[no-op edit]" and the model fell back to blind sed). Trust
            # the model's whitespace verbatim.
            return _apply_edit(path, data, data[:s] + raw + data[e:],
                               " (applied verbatim: whitespace-only change)"
                               + (note_new if used_unew else ""))
        return _apply_edit(path, data, data[:s] + repl + data[e:],
                           " (recovered: matched ignoring indentation/whitespace)"
                           + (note_new if used_unew else ""))
    if len(spans) > 1:
        return (f"[old string matches {len(spans)} places ignoring whitespace; include "
                f"more surrounding lines to make it unique]")

    return (f"[old string not found; no change made.{_closest_hint(data, old)}]"
            + _indent_hint(data, old))


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


def _fit_indent(new: str, target_indent: str) -> tuple[str, bool]:
    """Slide the whole `new` block so its first non-blank line carries `target_indent`,
    preserving every line's indentation RELATIVE to that first line (dedents included).

    This is what lets replace_lines take the indentation burden off the model: it can
    send the block at any base column — flush left, or copied verbatim from a numbered
    read — and we shift it to where it lands. Returns (fitted, shifted); shifted is False
    and `new` is returned untouched when the block is already at the target column, has no
    non-blank line, OR the target is tab-indented (a char-delta shift would emit spaces and
    mix them with tabs — the unit-aware recoveries in _splice handle that case)."""
    lines = new.split("\n")
    first = next((l for l in lines if l.strip()), None)
    if first is None or "\t" in target_indent:
        return new, False
    delta = len(target_indent) - (len(first) - len(first.lstrip()))
    if delta == 0:
        return new, False
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        else:
            cur = len(ln) - len(ln.lstrip())
            out.append(" " * max(0, cur + delta) + ln.lstrip())
    return "\n".join(out), True


def _snap_indent(new: str, target_indent: str) -> str:
    """Force EVERY non-blank line to `target_indent`, dropping the model's own relative
    indentation. The recovery for a UNIFORM-level block the model wrote at inconsistent
    columns (e.g. sibling class fields). `target_indent` carries the file's own whitespace
    (tabs or spaces), so this is unit-correct. A genuinely nested block would be flattened,
    so callers accept the snap ONLY when the result parses (else the indent-reject stands)."""
    return "\n".join((target_indent + ln.lstrip()) if ln.strip() else ""
                     for ln in new.split("\n"))


def _scan_py_line(line: str, in_str: str | None, depth: int) -> tuple[str | None, int, bool]:
    """Scan ONE physical line of Python, starting in the given state, char by char.
    Returns (in_str_after, bracket_depth_after, opens_block):
      in_str  — the active triple-quote delimiter (\"\"\" or ''') mid-string, else None;
      depth   — running (), [], {} nesting depth (continuation when > 0);
      opens_block — the logical line ended (outside strings/brackets) with ':'.
    Correct string tracking is the safety-critical part: it keeps _reindent_python from
    ever touching bytes inside a triple-quoted string."""
    i, n, last = 0, len(line), ""
    while i < n:
        c = line[i]
        if in_str is not None:
            if line.startswith(in_str, i):
                in_str, i = None, i + 3
            elif c == "\\":
                i += 2
            else:
                i += 1
            continue
        if c in "\"'":
            if line[i:i + 3] in ('"""', "'''"):
                in_str, i = line[i:i + 3], i + 3
                continue
            j = i + 1                              # single-line string: skip to its close
            while j < n:
                if line[j] == "\\":
                    j += 2
                    continue
                if line[j] == c:
                    break
                j += 1
            i, last = j + 1, "s"
            continue
        if c == "#":
            break                                  # comment runs to end of line
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        if not c.isspace():
            last = c
        i += 1
    return in_str, depth, (in_str is None and depth == 0 and last == ":")


_DEDENT_KW = ("else", "elif", "except", "finally")


def _reindent_python(new: str, base_indent: str, unit: str) -> str:
    """Recompute a Python block's indentation from its OWN syntax, ignoring the whitespace
    the model sent (which a weak model gets wrong in BOTH directions — over-indenting a
    comment, under-indenting a body). Rules that don't trust the model's columns:
      • a ':'-terminated line opens a block, and the NEXT logical line is Python's REQUIRED
        body, so it is forced one level deeper whatever width the model gave it;
      • a line that starts with else/elif/except/finally closes one block (dedents);
      • otherwise a line dedents by however many open blocks its model width fell below —
        the one place the model's RELATIVE width is used, and only to leave blocks.
    Lines inside a triple-quoted string and bracket-continuation lines are copied VERBATIM
    (never reindented), so string contents can't be corrupted. Best-effort: the caller
    applies the result ONLY if the file then parses, so a mis-read block falls through."""
    out: list[str] = []
    stack: list[int] = []          # body-ref model widths, one per open block; len == level
    in_str: str | None = None
    depth = 0
    pending_body = False           # previous logical line ended with ':' → this line is its body
    for ln in new.split("\n"):
        if in_str is not None:                     # inside a multi-line string → verbatim
            out.append(ln)
            in_str, depth, _ = _scan_py_line(ln, in_str, depth)
            continue
        stripped = ln.strip()
        if not stripped:
            out.append("")
            continue
        if depth > 0:                              # bracket continuation → keep, indent past block
            out.append(base_indent + unit * (len(stack) + 1) + stripped)
            in_str, depth, _ = _scan_py_line(ln, in_str, depth)
            continue
        model_w = len(ln) - len(ln.lstrip())
        if pending_body:
            stack.append(model_w)                  # the required body defines this block's level
            pending_body = False
        else:
            while stack and model_w < stack[-1]:   # model shows this line left inner block(s)
                stack.pop()
            kw = re.match(r"\w+", stripped)         # else/elif/except/finally close one block
            if kw and kw.group() in _DEDENT_KW and stack:
                stack.pop()
        out.append(base_indent + unit * len(stack) + stripped)
        in_str, depth, opens = _scan_py_line(ln, in_str, depth)
        if opens:
            pending_body = True
    return "\n".join(out)


def _splice(path: str, data: str, prefix: str, suffix: str, new: str,
            target_indent: str, ended_nl: bool, label: str) -> str:
    """Fit `new` to `target_indent`, splice between prefix/suffix, apply via _apply_edit.
    If the fitted result would newly break Python indentation, try two recoveries in order,
    each accepted only when it stops the indent break: (1) a structural reindent that
    recomputes levels from the block's syntax (fixes a multi-LEVEL block the model
    mis-indented), then (2) a uniform snap (fixes a single-level block at inconsistent
    columns). If neither parses, the fitted attempt lands so the model gets the reject +
    steer. Shared by replace_lines and insert_lines."""
    def shape(t: str) -> str:
        # Match the boundary's trailing-newline shape: terminate the block mid-file so the
        # following line stays put; leave it unterminated only at a no-newline EOF.
        if ended_nl and not t.endswith("\n"):
            return t + "\n"
        if not ended_nl and t.endswith("\n"):
            return t.rstrip("\n")
        return t
    fitted, shifted = _fit_indent(new, target_indent)
    after = prefix + shape(fitted) + suffix
    note = f" ({label}{'; fit indentation' if shifted else ''})"
    if not syntaxgate.indent_reject(path, data, after):
        return _apply_edit(path, data, after, note)
    # Fitted breaks indentation — recover. Each candidate is gated by indent_reject, which
    # only clears when the file parses without a NEW IndentationError, so a bad recovery
    # can never land; we just fall through to the next.
    unit = _indent_unit(data)
    if (levers.enabled("structural_reindent")
            and repomap.service().lang_for(path) == "python"):
        cand = prefix + shape(_reindent_python(new, target_indent, unit)) + suffix
        if not syntaxgate.indent_reject(path, data, cand):
            return _apply_edit(path, data, cand, f" ({label}; reindented to structure)")
    cand = prefix + shape(_snap_indent(new, target_indent)) + suffix
    if not syntaxgate.indent_reject(path, data, cand):
        return _apply_edit(path, data, cand, f" ({label}; snapped indentation)")
    return _apply_edit(path, data, after, note)


def tool_replace_lines(path: str, start: int, end: int, new: str) -> str:
    """Replace file lines [start, end] (1-based, inclusive — the numbers `read` prints)
    with `new`, fitting the new block's indentation to the region it lands in. This is
    the line-addressed alternative to `edit`: the model that already knows the line
    numbers doesn't re-quote an exact `old` string and doesn't re-transcribe leading
    whitespace (the two things that drive the string-edit death loop). Empty `new` deletes
    the range. Goes through _apply_edit, so the same indent-reject + syntax gate apply."""
    if not os.path.exists(path) and path.startswith("/") and os.path.exists(path.lstrip("/")):
        path = path.lstrip("/")
    if not os.path.exists(path):
        return f"[no such file: {path}]"
    if isinstance(start, bool) or isinstance(end, bool) \
            or not isinstance(start, int) or not isinstance(end, int):
        return "[replace_lines: start and end must be integer line numbers (1-based)]"
    with open(path, errors="replace") as f:
        # readlines() (each line keeps its trailing "\n") so our line numbers match the
        # ones `read` prints — split("\n") would invent a phantom last line for a
        # newline-terminated file and clamping to it would eat the trailing newline.
        lines = f.readlines()
    data = "".join(lines)
    n = len(lines)
    if start < 1 or end < start:
        return (f"[replace_lines: invalid range start={start} end={end}; need "
                f"1 <= start <= end (1-based, inclusive).]")
    if start > n:
        return (f"[replace_lines: start={start} is past the last line ({n}); append with "
                f"write, or pick a start within the file.]")
    end = min(end, n)
    prefix, replaced, suffix = ("".join(lines[:start - 1]),
                                "".join(lines[start - 1:end]),
                                "".join(lines[end:]))
    if new == "":
        return _apply_edit(path, data, prefix + suffix, f" (deleted lines {start}-{end})")
    anchor = next((ln for ln in replaced.split("\n") if ln.strip()), "")
    target_indent = anchor[: len(anchor) - len(anchor.lstrip())]
    return _splice(path, data, prefix, suffix, new, target_indent,
                   replaced.endswith("\n"), f"replaced lines {start}-{end}")


def tool_insert_lines(path: str, after_line: int, code: str) -> str:
    """Insert `code` as new line(s) immediately AFTER line `after_line` (1-based; 0 = the
    very top of the file), inheriting that line's indentation so you never supply a column.
    The line-addressed complement to replace_lines for ADDING a field/statement beside a
    sibling — the case no symbol tool covers cleanly (a dataclass field or a bare statement
    isn't a function/class symbol). Same fit + snap + indent-reject path as replace_lines."""
    if not os.path.exists(path) and path.startswith("/") and os.path.exists(path.lstrip("/")):
        path = path.lstrip("/")
    if not os.path.exists(path):
        return f"[no such file: {path}]"
    if isinstance(after_line, bool) or not isinstance(after_line, int):
        return "[insert_lines: after_line must be an integer line number (1-based; 0 = top)]"
    if code == "":
        return "[insert_lines: code is empty; nothing to insert]"
    with open(path, errors="replace") as f:
        lines = f.readlines()
    data = "".join(lines)
    n = len(lines)
    if after_line < 0 or after_line > n:
        return (f"[insert_lines: after_line={after_line} out of range; the file has {n} "
                f"lines (use 0 to insert at the top).]")
    anchor = (lines[after_line - 1] if after_line >= 1 else (lines[0] if lines else "")).rstrip("\n")
    target_indent = anchor[: len(anchor) - len(anchor.lstrip())]
    prefix = "".join(lines[:after_line])
    suffix = "".join(lines[after_line:])
    # Inserting right after a block opener (a line ending in ':') means the new code
    # belongs to that block's BODY, one level deeper — inherit the existing body's indent
    # when there is one, else add four spaces to the opener's. Sibling inserts (a field
    # after a field) don't end in ':', so they keep the anchor's own indent.
    if anchor.rstrip().endswith(":"):
        body = next((ln for ln in suffix.split("\n") if ln.strip()), "")
        body_indent = body[: len(body) - len(body.lstrip())]
        target_indent = body_indent if len(body_indent) > len(target_indent) \
            else target_indent + "    "
    # Start the insert on its own line. prefix lacks a trailing newline only when we insert
    # after an unterminated final line (after_line == n at a no-newline EOF): add one, and
    # keep the file's no-trailing-newline shape for the new last line.
    if prefix and not prefix.endswith("\n"):
        prefix, ended_nl = prefix + "\n", False
    else:
        ended_nl = True
    return _splice(path, data, prefix, suffix, code, target_indent, ended_nl,
                   f"inserted after line {after_line}")


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


def _walk_glob(path: str, pattern: str):
    """Pruned-walk equivalent of `glob(os.path.join(path, pattern), recursive=True)`
    for basename-only patterns — "**/*" or "**/<name-glob>" — which is what the model
    actually sends. glob materializes the ENTIRE tree (weights dirs, node_modules, VCS
    caches included) and only then lets `_skip` filter; the walk prunes those dirs
    before descending, keeps glob's hidden-file rule, and doesn't follow dir symlinks.
    Returns None for structured patterns (a "/" or "**" past the leading "**/") so
    callers fall back to glob. Yields dirs too, matching glob("**/*")."""
    if pattern in ("**", "**/*"):
        base = "*"
    elif pattern.startswith("**/") and "/" not in pattern[3:] and "**" not in pattern[3:]:
        base = pattern[3:]
    else:
        return None

    def walk():
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in IGNORE_DIRS]
            for d in dirnames:
                if fnmatch.fnmatch(d, base):
                    yield os.path.join(dirpath, d)
            for fn in filenames:
                if not fn.startswith(".") and fnmatch.fnmatch(fn, base):
                    yield os.path.join(dirpath, fn)

    return walk()


def tool_glob(pattern: str, should_stop=None) -> str:
    it = _walk_glob(".", pattern)
    if it is None:
        hits = [h for h in sorted(_glob.glob(pattern, recursive=True)) if not _skip(h)]
    else:
        # walk yields "./x/y" but glob(pattern) yields "x/y" — keep the output stable
        hits = sorted(h[2:] if h.startswith("./") else h for h in it)
    return "\n".join(hits[:200]) or "[no matches]"


# grep output budgets. A single match in a minified JS/JSON file dumps a multi-MB
# line straight into the transcript — the same blowup READ_MAX_CHARS guards on the
# read path, and every char is prefill on a ~350 tok/s model. So cap each emitted
# line, the total number of lines, and the number of files walked — and, unlike the
# old code, ANNOUNCE when a cap binds so the model narrows the query instead of
# concluding "no matches".
GREP_MAX_LINES = 200
GREP_MAX_FILES = 5000
GREP_LINE_CHARS = 500
# Total output cap (mirrors READ_MAX_CHARS). The line cap alone isn't enough: 200 lines
# × 500 chars = ~100k chars ≈ 28k tokens of prefill. Dogfooding a "find every CHAD_ use"
# grep returned 84 long match lines = ~6k tokens = a ~19s stall before the next turn.
# ~10k chars (≈3k tokens) keeps a wide grep responsive; the truncation notice already
# tells the model to narrow the pattern or add a path for the rest.
GREP_MAX_CHARS = 10000


def _grep_clip(s: str) -> str:
    return s if len(s) <= GREP_LINE_CHARS else s[:GREP_LINE_CHARS] + "…[line clipped]"


# Files up to this size are read whole and prescreened with one C-speed regex pass;
# only files that contain a match pay the (slow) per-line Python loop. Bigger files
# keep the old streaming scan.
GREP_FULLREAD_MAX = 4 * 1024 * 1024


def _grep_prescreen_rx(pattern: str, flags: int):
    """A whole-file version of the per-line search: re.MULTILINE makes ^/$ anchor per
    line, so any line the per-line loop would match, this finds somewhere in the full
    text (a superset — false positives just run the line loop and emit nothing). The
    exceptions where a full-text search could MISS a per-line match: \\A/\\Z (different
    meaning across the two scans) and negative lookarounds (can see past the line's
    \\n and reject). Those patterns return None: no prescreen, stream as before."""
    if any(tok in pattern for tok in (r"\A", r"\Z", "(?!", "(?<!")):
        return None
    try:
        return re.compile(pattern, flags | re.MULTILINE)
    except re.error:
        return None


def tool_grep(pattern: str, path: str = ".", glob: str = "**/*", ignore_case: bool = False,
              context: int = 0, should_stop=None) -> str:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as rx_err:
        return f"[bad regex: {rx_err}]"
    rx_pre = _grep_prescreen_rx(pattern, flags)
    ctx = max(0, min(int(context or 0), 5))
    out: list[str] = []
    out_chars = 0      # running total of emitted chars (bounds prefill; see GREP_MAX_CHARS)
    matches = 0        # total match lines seen (may exceed what we emit once capped)
    capped = False     # hit the line- or char-count output cap

    def emit(s: str):
        nonlocal capped, out_chars
        if len(out) < GREP_MAX_LINES and out_chars < GREP_MAX_CHARS:
            out.append(s)
            out_chars += len(s) + 1  # +1 for the join newline
        else:
            capped = True

    # The model routinely passes a file as `path` (the schema says directory); walking
    # a file yields nothing, which used to read as a clean "[no matches]". Search the
    # named file instead — explicit naming also overrides the skip list, like `read`.
    explicit_file = os.path.isfile(path)
    if explicit_file:
        files, files_truncated = [path], False
    elif not os.path.isdir(path):
        return f"[path not found: {path}]"
    else:
        # Count only files we would actually SEARCH against GREP_MAX_FILES — apply the
        # dir/skip filter BEFORE the cap, not after. The demonstrated starvation
        # (django-16454): _walk_glob yields directories too, and Django's locale tree is
        # mostly dirs + skipped blobs, so the 5000-slot budget was exhausted on entries
        # that can never match before the walk ever reached django/core/, and the target
        # symbol (position ~3,195 among real files) sat in an unsearched file. Filtering
        # first roughly doubles the effective reach on a dir-heavy tree.
        fast = _walk_glob(path, glob)
        if fast is None:
            fast = _glob.glob(os.path.join(path, glob), recursive=True)
        files, files_truncated = [], False
        filter_first = levers.enabled("grep_filter_before_cap")
        for fp in fast:
            if filter_first and (_skip(fp) or not os.path.isfile(fp)):
                continue
            if len(files) >= GREP_MAX_FILES:
                files_truncated = True
                break
            files.append(fp)
        if not filter_first:  # pre-iter-3: dirs/skipped blobs consumed the budget
            files = [fp for fp in files if not _skip(fp) and os.path.isfile(fp)]
    for fp in files:
        if should_stop and should_stop():
            return "[interrupted by user]"
        if not explicit_file and (_skip(fp) or not os.path.isfile(fp)):
            continue
        try:
            # Prescreen: one full-text regex pass (C speed) decides whether the file
            # is worth the per-line Python loop at all. io.StringIO(text) then feeds
            # that loop the exact same lines an open file would, so match/emit
            # semantics are unchanged.
            text = None
            if rx_pre is not None and os.path.getsize(fp) <= GREP_FULLREAD_MAX:
                with open(fp, errors="ignore") as f:
                    text = f.read()
                if not rx_pre.search(text):
                    continue
            src: io.TextIOBase = (io.StringIO(text) if text is not None
                                  else open(fp, errors="ignore"))
            if ctx:
                # context mode needs surrounding lines, so read the whole file and
                # emit `--`-separated groups (merging overlapping windows), like grep -C.
                with src as fh:
                    flines = [ln.rstrip("\n") for ln in fh]
                idxs = [i for i, ln in enumerate(flines) if rx.search(ln)]
                if not idxs:
                    continue
                matches += len(idxs)
                groups: list[list[int]] = []  # merged [start, end] inclusive windows
                for i in idxs:
                    s, e = max(0, i - ctx), min(len(flines) - 1, i + ctx)
                    if groups and s <= groups[-1][1] + 1:
                        groups[-1][1] = max(groups[-1][1], e)
                    else:
                        groups.append([s, e])
                for gi, (s, e) in enumerate(groups):
                    if gi:
                        emit("--")
                    for j in range(s, e + 1):
                        sep = ":" if rx.search(flines[j]) else "-"
                        emit(f"{fp}:{j+1}{sep} {_grep_clip(flines[j])}")
            else:
                with src as fh:
                    for i, ln in enumerate(fh, 1):
                        if rx.search(ln):
                            matches += 1
                            emit(f"{fp}:{i}: {_grep_clip(ln.rstrip())}")
        except OSError:
            continue

    if not out:
        # State the searched scope, and NEVER hide truncation on the zero-match path:
        # a capped walk that returns a bare "[no matches]" is a confident lie — the
        # demonstrated failure (django-14007): the tree exceeded GREP_MAX_FILES, the
        # issue's own symbol lived in an unsearched file, and the model stalled out
        # trusting the empty result.
        if not levers.enabled("grep_zero_match_notice"):
            return "[no matches]"  # pre-iter-2: the confident lie, for the ablation arm
        scope = path if path not in (".", "") else "the current directory"
        msg = f"[no matches for {pattern!r} in {scope}]"
        if files_truncated:
            msg += (f"\n[WARNING: only the first {GREP_MAX_FILES} files were searched — "
                    "this tree is larger, so the pattern may exist in files that were "
                    "not searched. Re-run with a narrower path= (e.g. the package "
                    "subdirectory).]")
        return msg
    notices = []
    if capped:
        notices.append(f"[results truncated: {len(out)}/{matches} lines — narrow the "
                       f"pattern or add a path]")
    if files_truncated:
        notices.append(f"[searched first {GREP_MAX_FILES} files]")
    return "\n".join(out + notices)


# Each entry takes (args, should_stop); long-running tools honor should_stop so a
# ctrl-c interrupt can abort them mid-flight.
DISPATCH = {
    "write_todos": lambda a, ss=None: tool_write_todos(a["todos"]),
    "bash": lambda a, ss=None: tool_bash(a["command"], a.get("timeout", 120), should_stop=ss),
    "read": lambda a, ss=None: tool_read(a["path"], a.get("offset", 0),
                                         a.get("limit", READ_DEFAULT_LIMIT)),
    "write": lambda a, ss=None: tool_write(a["path"], a["content"]),
    "edit": lambda a, ss=None: tool_edit(a["path"], a["old"], a["new"]),
    "replace_lines": lambda a, ss=None: tool_replace_lines(
        a["path"], a["start"], a["end"], a["new"]),
    "insert_lines": lambda a, ss=None: tool_insert_lines(
        a["path"], a["after_line"], a["code"]),
    "glob": lambda a, ss=None: tool_glob(a["pattern"], should_stop=ss),
    "grep": lambda a, ss=None: tool_grep(a["pattern"], a.get("path", "."),
                                         a.get("glob", "**/*"),
                                         a.get("ignore_case", False),
                                         a.get("context", 0), should_stop=ss),
    # Symbolic code tools. READS go through the tree-sitter backend (repomap) — it's
    # language-agnostic and the repo_map gives a ranked skeleton for cheap navigation.
    # EDITS stay on the jedi backend (symbols), the proven Python symbol editor.
    "repo_map": lambda a, ss=None: repomap.service().repo_map(
        a.get("budget", 1500), a.get("focus"), should_stop=ss),
    "overview": lambda a, ss=None: repomap.service().overview(a["path"], should_stop=ss),
    "view_symbol": lambda a, ss=None: repomap.service().view_symbol(
        a["name"], a.get("path"), should_stop=ss),
    "find_symbol": lambda a, ss=None: repomap.service().find_symbol(a["name"], should_stop=ss),
    "find_refs": lambda a, ss=None: repomap.service().find_refs(
        a["name"], a.get("path"), should_stop=ss),
    "replace_symbol": lambda a, ss=None: symbols.service().replace_symbol(
        a["name"], a["new"], a.get("path"), should_stop=ss),
    "insert_symbol": lambda a, ss=None: symbols.service().insert_symbol(
        a["name"], a["code"], a.get("where", "after"), a.get("path"), should_stop=ss),
    "rename_symbol": lambda a, ss=None: repomap.service().rename_symbol(
        a["name"], a["new_name"], a.get("path"), should_stop=ss),
    # Agent Skills (https://agentskills.io): load one skill's full instructions on
    # demand (tier-2 progressive disclosure). Registered in active_schemas() only when
    # skills are installed; the dispatch is harmless (a clear message) otherwise.
    "activate_skill": lambda a, ss=None: _skills().activate(a["name"]),
}


def _skills():
    """Lazy import of the skills module (avoids a circular import at module load)."""
    from . import skills
    return skills

# Tools that mutate state -> CLI asks for confirmation unless --yolo.
MUTATING = {"bash", "write", "edit", "replace_lines", "insert_lines", "replace_symbol",
            "insert_symbol", "rename_symbol"}

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
            "name": "task",
            "description": "Delegate an open-ended exploration or sub-task to a fresh "
                           "sub-agent that works in its OWN small, isolated context and "
                           "returns only a condensed answer. Use this for spelunking — "
                           "'find where X is handled', 'which files touch Y', 'trace how "
                           "Z flows' — so your MAIN context stays small and cheap (the "
                           "sub-agent's grep/read churn never enters this conversation; "
                           "only its final findings do). Read-only by default. The "
                           "sub-agent does NOT see this conversation, so put everything it "
                           "needs in `prompt`. It cannot spawn further sub-agents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string",
                                    "description": "A short (3-6 word) label for the sub-task."},
                    "prompt": {"type": "string",
                               "description": "The full, self-contained instruction: what to "
                                              "find or do, and exactly what to report back."},
                    "tools": {"type": "string", "enum": ["read-only", "all"],
                              "description": "Tool access: 'read-only' (default; search/read "
                                             "only) or 'all' (also edit/run — honored only in "
                                             "--yolo/auto mode; otherwise clamped to read-only)."},
                },
                "required": ["description", "prompt"],
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
            "name": "read",
            "description": ("Read a text file with line numbers. Returns up to "
                            f"{READ_DEFAULT_LIMIT} lines by default ({READ_MAX_LIMIT} max) "
                            "to keep context small. A default read of a large code file "
                            f"(>{READ_SKELETON_LINES} lines) returns its STRUCTURE "
                            "(signatures) instead of the body — then use view_symbol(name) "
                            "for a function's body, or read(path, offset=N) to page raw lines."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "Start line (0-based)."},
                    "limit": {"type": "integer",
                              "description": f"Max lines (capped at {READ_MAX_LIMIT})."},
                },
                "required": ["path"],
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
                           "an EXACT match of `old` including indentation; when you already "
                           "know the line numbers (from read), prefer replace_lines, which "
                           "doesn't make you re-quote the text or its whitespace.",
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
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": "Replace a RANGE OF LINES, addressed by the line numbers `read` "
                           "prints, with new text. This is the reliable way to edit a region "
                           "you've already located: give the 1-based start and end (inclusive) "
                           "and the replacement — you do NOT need to re-quote the old text or "
                           "match its exact leading whitespace, because the new block's "
                           "indentation is fitted to the target automatically. Pass an empty "
                           "`new` to delete the lines. Prefer this over `edit` whenever you "
                           "know the line numbers, and over rewriting a whole function when "
                           "only a few lines change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer",
                              "description": "First line to replace (1-based, inclusive)."},
                    "end": {"type": "integer",
                            "description": "Last line to replace (1-based, inclusive)."},
                    "new": {"type": "string",
                            "description": "Replacement text (empty string deletes the range)."},
                },
                "required": ["path", "start", "end", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_lines",
            "description": "Insert new line(s) immediately AFTER a given line number (from "
                           "read), with the right indentation applied for you — so you do NOT "
                           "supply leading whitespace: a sibling's indent when you insert after "
                           "a normal line, or the block body's indent when you insert right "
                           "after a line ending in ':'. Use this to ADD a field, statement, "
                           "import, or case next to an existing one (e.g. a new dataclass field "
                           "beside a sibling field). Pass after_line=0 to insert at the very "
                           "top. For a whole new function/method, insert_symbol is better.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "after_line": {"type": "integer",
                                   "description": "Insert after this 1-based line (0 = top of file)."},
                    "code": {"type": "string", "description": "The line(s) to insert."},
                },
                "required": ["path", "after_line", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern (supports ** recursion).",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with a regex; returns path:line: match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string",
                             "description": "Directory or file to search (default '.')."},
                    "glob": {"type": "string", "description": "File glob (default '**/*')."},
                    "ignore_case": {"type": "boolean",
                                    "description": "Case-insensitive match (default false)."},
                    "context": {"type": "integer",
                                "description": "Lines of context to show before/after each "
                                               "match, 0-5 (default 0)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_map",
            "description": "Get a ranked, signatures-only map of the whole codebase (most "
                           "referenced files first; functions/classes with line numbers, no "
                           "bodies). Call this FIRST on an unfamiliar project to orient yourself "
                           "cheaply instead of reading files — it costs a few hundred tokens for "
                           "the entire repo. Then use find_symbol/view_symbol to drill in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "integer",
                               "description": "Approx token budget for the map (default 1500)."},
                    "focus": {"type": "array", "items": {"type": "string"},
                              "description": "Optional path substrings to rank toward."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "overview",
            "description": "List the functions and classes defined in ONE file (names, "
                           "signatures, line numbers) WITHOUT their bodies — any language. Use "
                           "this to understand a file cheaply instead of reading the whole thing.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_symbol",
            "description": "Show the full source of ONE function/class/method by name, instead "
                           "of reading an entire file. Name may be qualified ('Class/method') to "
                           "disambiguate. Prefer this over `read` for inspecting code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol or 'Class/method'."},
                    "path": {"type": "string", "description": "Optional file to disambiguate."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "Find where a function/class/method is DEFINED across the project "
                           "(returns path:line + signature). Use this instead of grep to locate "
                           "a definition.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_refs",
            "description": "Find every place a symbol is USED across the project — precise "
                           "(a real language server follows imports and scope, so it won't "
                           "confuse a method with an unrelated function of the same name; far "
                           "better than grep before a rename/refactor). Name may be "
                           "'Class/method'; pass path to disambiguate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string", "description": "Optional file to disambiguate."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_symbol",
            "description": "Replace the ENTIRE source of one function/class/method with new code "
                           "(found by name, not text matching — robust to whitespace). `new` is "
                           "the complete new definition including its signature and indentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol or 'Class/method'."},
                    "new": {"type": "string", "description": "Complete replacement source."},
                    "path": {"type": "string", "description": "Optional file to disambiguate."},
                },
                "required": ["name", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_symbol",
            "description": "Insert new code immediately before or after an existing symbol "
                           "(e.g. add a new function/method next to a related one).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Anchor symbol or 'Class/method'."},
                    "code": {"type": "string", "description": "New code to insert."},
                    "where": {"type": "string", "enum": ["after", "before"],
                              "description": "Default 'after'."},
                    "path": {"type": "string", "description": "Optional file to disambiguate."},
                },
                "required": ["name", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_symbol",
            "description": "Rename a function/class/method AND every precise reference to it "
                           "across the project in one step — the safe way to do a multi-file "
                           "rename. Uses a real language server, so it follows imports and "
                           "scope and will NOT touch an unrelated symbol of the same name; "
                           "each identifier is rewritten by position, never by text match. "
                           "Pass path to disambiguate when several symbols share the name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Current symbol or 'Class/method'."},
                    "new_name": {"type": "string", "description": "New identifier."},
                    "path": {"type": "string", "description": "Optional file to disambiguate."},
                },
                "required": ["name", "new_name"],
            },
        },
    },
]

# A/B knob (used by the eval harness): with CHAD_NO_SYMBOLS set, hide the
# tree-sitter symbolic tools so the agent must navigate with read/grep/glob only.
# Lets us measure the prefill savings the symbolic/repo-map layer actually buys.
#
# Read the env PER RENDER (not once at import) so a single loaded engine can serve
# both arms of `run_evals.py --ab` in-process: the harness flips CHAD_NO_SYMBOLS
# between arms and the agent's render path calls active_schemas() each turn. SCHEMAS
# itself stays the full list (name lookups / required-arg validation need every tool).
_SYMBOLIC = {"repo_map", "overview", "view_symbol", "find_symbol", "find_refs",
             "replace_symbol", "insert_symbol", "rename_symbol"}


def _activate_skill_schema(names):
    """The activate_skill tool schema, with `name` constrained to the set of installed
    skill names so the model can't hallucinate one that doesn't exist (spec guidance)."""
    return {
        "type": "function",
        "function": {
            "name": "activate_skill",
            "description": "Load the full instructions for one of the available skills "
                           "(listed under '# Skills' in the system prompt) before doing a "
                           "task that matches its description. Returns the skill's "
                           "step-by-step instructions and the files it bundles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": names,
                             "description": "Exact name of the skill to activate."},
                },
                "required": ["name"],
            },
        },
    }


def active_schemas():
    """The tool schemas to expose to the model right now. Starts from SCHEMAS (minus the
    symbolic tools when CHAD_NO_SYMBOLS is set) and appends the activate_skill tool when
    any Agent Skills are installed for the current project/user (omitted otherwise, so a
    skill-less project never sees a dead tool)."""
    schemas = SCHEMAS
    # Subagent/Task tool (plan 041) ships opt-out: CHAD_NO_TASK hides it (the A/B arm and
    # the escape hatch if the model misuses it). The subagent's OWN render drops it again
    # via Agent._active_schemas — reentrancy guard, subagents can't spawn subagents.
    if config.flag("CHAD_NO_TASK"):
        schemas = [s for s in schemas if s["function"]["name"] != "task"]
    if config.flag("CHAD_NO_SYMBOLS"):
        schemas = [s for s in schemas if s["function"]["name"] not in _SYMBOLIC]
    names = _skills().skill_names()
    if names:
        schemas = schemas + [_activate_skill_schema(names)]
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
