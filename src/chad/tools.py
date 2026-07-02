"""Claude-Code-style tools for the agent loop.

Each tool has an OpenAI/Qwen-compatible JSON schema (exposed to the model via the
chat template's `tools` argument) and a Python implementation. Implementations are
deliberately conservative: reads are unrestricted, writes/bash are real but the CLI
gates them behind a confirmation unless --yolo is set.
"""

import glob as _glob
import os
import re
import signal
import subprocess
import threading
import time
from typing import Any

from . import repomap, symbols, syntaxgate
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
READ_MAX_CHARS = 24000
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
            return (f"[{_rel(path)} is {total} lines — showing structure only to keep "
                    f"context small. Use view_symbol(name) for one symbol's body, or "
                    f"read(path, offset=N) to page raw lines.]\n{skel}")

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


def _reindent(new: str, target_indent: str) -> str:
    """Shift `new` so its first non-blank line carries target_indent, preserving the
    relative indentation of the rest (so a recovered block lands at the file's indent)."""
    lines = new.split("\n")
    first = next((l for l in lines if l.strip()), "")
    src = first[: len(first) - len(first.lstrip())]
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
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
            f"indentation), or use replace_symbol to rewrite the whole function.") if best else ""


def _apply_edit(path: str, before: str, after: str, note: str) -> str:
    if after == before:
        return "[no-op edit: the replacement leaves the file unchanged]"
    with open(path, "w") as f:
        f.write(after)
    result = f"[edited {_rel(path)}{note}]"
    warn = syntaxgate.check_syntax(path, before)
    return result + warn if warn else result


def tool_edit(path: str, old: str, new: str) -> str:
    if not os.path.exists(path):
        return f"[no such file: {path}]"
    if old == new:
        return "[no-op edit: old and new are identical; change the content or stop]"
    with open(path) as f:
        data = f.read()

    # (1) exact match — the common, fast path.
    n = data.count(old)
    if n == 1:
        return _apply_edit(path, data, data.replace(old, new, 1), "")
    if n > 1:
        return f"[old string appears {n} times; make it unique by including more surrounding lines]"

    # (2) escape-normalized: literal \n/\t in `old` (and `new` when it has no real newline).
    uold = _unescape_ws(old)
    unew = _unescape_ws(new) if ("\n" not in new and "\\n" in new) else new
    if uold != old:
        c = data.count(uold)
        if c == 1:
            return _apply_edit(path, data, data.replace(uold, unew, 1),
                               " (recovered: interpreted \\n/\\t escapes in `old`)")
        if c > 1:
            return f"[old string appears {c} times; make it unique by including more surrounding lines]"

    # (3) whitespace-flexible: indentation / trailing-space drift, still requiring uniqueness.
    probe = uold if uold != old else old
    spans = _ws_flexible_spans(data, probe)
    if len(spans) == 1:
        s, e = spans[0]
        head = data[s:e].split("\n")[0]
        indent = head[: len(head) - len(head.lstrip())]
        repl = _reindent((unew if uold != old else new).strip("\n"), indent)
        return _apply_edit(path, data, data[:s] + repl + data[e:],
                           " (recovered: matched ignoring indentation/whitespace)")
    if len(spans) > 1:
        return (f"[old string matches {len(spans)} places ignoring whitespace; include "
                f"more surrounding lines to make it unique]")

    return f"[old string not found; no change made.{_closest_hint(data, old)}]"


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


def tool_glob(pattern: str, should_stop=None) -> str:
    hits = [h for h in sorted(_glob.glob(pattern, recursive=True)) if not _skip(h)]
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


def _grep_clip(s: str) -> str:
    return s if len(s) <= GREP_LINE_CHARS else s[:GREP_LINE_CHARS] + "…[line clipped]"


def tool_grep(pattern: str, path: str = ".", glob: str = "**/*", ignore_case: bool = False,
              context: int = 0, should_stop=None) -> str:
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as rx_err:
        return f"[bad regex: {rx_err}]"
    ctx = max(0, min(int(context or 0), 5))
    out: list[str] = []
    matches = 0        # total match lines seen (may exceed what we emit once capped)
    capped = False     # hit the GREP_MAX_LINES output cap

    def emit(s: str):
        nonlocal capped
        if len(out) < GREP_MAX_LINES:
            out.append(s)
        else:
            capped = True

    files = _glob.glob(os.path.join(path, glob), recursive=True)
    files_truncated = len(files) > GREP_MAX_FILES
    for fp in files[:GREP_MAX_FILES]:
        if should_stop and should_stop():
            return "[interrupted by user]"
        if _skip(fp) or not os.path.isfile(fp):
            continue
        try:
            if ctx:
                # context mode needs surrounding lines, so read the whole file and
                # emit `--`-separated groups (merging overlapping windows), like grep -C.
                with open(fp, errors="ignore") as f:
                    flines = [ln.rstrip("\n") for ln in f]
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
                with open(fp, errors="ignore") as f:
                    for i, ln in enumerate(f, 1):
                        if rx.search(ln):
                            matches += 1
                            emit(f"{fp}:{i}: {_grep_clip(ln.rstrip())}")
        except OSError:
            continue

    if not out:
        return "[no matches]"
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
MUTATING = {"bash", "write", "edit", "replace_symbol", "insert_symbol", "rename_symbol"}

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
                                             "only) or 'all' (also edit/run — use sparingly)."},
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
            "description": "Replace a unique substring in a file with new text.",
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
                    "path": {"type": "string", "description": "Directory to search (default '.')."},
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
    if os.environ.get("CHAD_NO_TASK"):
        schemas = [s for s in schemas if s["function"]["name"] != "task"]
    if os.environ.get("CHAD_NO_SYMBOLS"):
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
