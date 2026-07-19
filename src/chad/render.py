"""Terminal rendering for chad (extracted from agent.py).

Turns raw model tokens and tool results into clean display events, and renders a
compact, Claude-style activity view (a verb + target on one line, then a short
summary — line counts for reads, a +/- diff for edits, a few lines of bash output).
The `_emit(kind, text)` callback contract is the boundary: the plain REPL uses
`_default_emit` (colored stdout); the TUI passes its own emitter.

Pure presentation — no model, no agent state. `agent` re-exports the public names
(render_tool_start/render_tool_result/_default_emit/_StreamView/confirm_preview/
the C_* colors) so existing importers (tui.py, cli.py) keep working unchanged.
"""

import difflib
import json
import os
import re
import sys

C_DIM = "\033[2m"; C_CYAN = "\033[36m"; C_GREEN = "\033[32m"; C_YEL = "\033[33m"
C_RED = "\033[31m"; C_BOLD = "\033[1m"; C_RST = "\033[0m"

# Optional syntax highlighting. Pygments is a pure-Python OPTIONAL
# extra (`pip install 'chad[highlight]'`): when present, diff/preview code lines get
# per-token colors *within* the +/- line coloring; when absent, output is byte-identical
# to the un-highlighted path. Import-guarded so a bare install never fails, and gated so
# tests can force the plain path by flipping `_HAS_PYGMENTS`. Never run in the per-token
# streaming hot path — only in final diffs and confirm-preview bodies (see STOP notes).
try:
    from pygments import highlight as _pyg_highlight
    from pygments.formatters.terminal import TerminalFormatter as _PygTermFormatter
    from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
    from pygments.util import ClassNotFound as _PygClassNotFound
    _HAS_PYGMENTS = True
    _PYG_FMT = _PygTermFormatter()
except ImportError:  # pragma: no cover - exercised via monkeypatched _HAS_PYGMENTS
    _HAS_PYGMENTS = False


def _highlight_code(code: str, filename: str = "") -> str:
    """Return `code` with per-token ANSI colors when pygments is available, else the
    input unchanged (byte-identical). Best-effort: any lexer/format failure falls back
    to the plain text, so highlighting can never corrupt a diff line."""
    if not code or not _HAS_PYGMENTS:
        return code
    try:
        try:
            lexer = guess_lexer_for_filename(filename or "x.txt", code)
        except _PygClassNotFound:
            lexer = get_lexer_by_name("text")
        return _pyg_highlight(code, lexer, _PYG_FMT).rstrip("\n")
    except Exception:  # noqa: BLE001 — display path; never let highlighting raise
        return code


def confirm_preview(name: str, args: dict, max_lines: int = 6) -> str:
    """A short, human-readable summary of what a mutating tool will do, shown before
    the y/n approval. Bounded to a few lines so it never floods the prompt."""
    def clip(s, n=400):
        s = str(s)
        return s if len(s) <= n else s[:n] + " …"
    def head(s):
        lines = str(s).splitlines() or [str(s)]
        body = "\n".join("    " + clip(ln) for ln in lines[:max_lines])
        if len(lines) > max_lines:
            body += f"\n    … (+{len(lines) - max_lines} more lines)"
        return body
    if name == "bash":
        return clip(args.get("command", ""))
    if name == "write":
        return f"{args.get('path','?')}\n{head(args.get('content',''))}"
    if name == "edit":
        return (f"{args.get('path','?')}\n  - {clip(args.get('old',''))}"
                f"\n  + {clip(args.get('new',''))}")
    if name in ("replace_symbol", "insert_symbol"):
        body = args.get("new") or args.get("code") or ""
        loc = args.get("name", "?") + (f" in {args['path']}" if args.get("path") else "")
        return f"{loc}\n{head(body)}"
    if name == "rename_symbol":
        loc = f" in {args['path']}" if args.get("path") else ""
        return f"{args.get('name','?')} → {args.get('new_name','?')}{loc}"
    if name.startswith("mcp__"):
        # An MCP tool can do anything (write files, hit an API, send a message); show
        # its full arguments so the approval is informed.
        return head(_compact_args(args))
    # fallback: the old behavior
    return str(args.get("command") or args.get("path") or "")


def _compact_args(args) -> str:
    """A compact, single-string view of a tool's arguments for display/preview."""
    if not isinstance(args, dict):
        return str(args)
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)


# ---------------------------------------------------------------------------
# Streaming view: turn raw model tokens into clean display events.
# The model interleaves reasoning (<think>…</think>) and tool-call syntax
# (<tool_call>…</tool_call>, <function=…>) with its actual prose. We never want
# to dump tool-call JSON into the transcript, and reasoning should drive a
# "Thinking…" indicator rather than a wall of tokens. This recomputes the
# cleaned view each token and emits only the new suffix, so tags that split
# across token boundaries are handled correctly.
# ---------------------------------------------------------------------------

_TOOL_OPENERS = ("<tool_call>", "<function=")
_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>")


def _safe_cut(text: str) -> str:
    """Hold back a trailing fragment that might be the start of a tag we filter,
    so we never emit a half-formed '<thi' or '</tool_c' to the screen."""
    i = text.rfind("<")
    if i == -1:
        return text
    tail = text[i:]
    if any(tag != tail and tag.startswith(tail) for tag in _TAGS):
        return text[:i]
    return text


def _split_view(raw: str, final: bool, started_in_think: bool = False):
    """Return (prose, think) for the raw text so far. Reasoning becomes think text;
    tool-call syntax is dropped from prose entirely.

    With thinking enabled the chat template opens <think> in the prompt, so the
    stream starts inside reasoning and only emits the closing </think> — there is
    no opening tag to match. `started_in_think` handles that: the leading text up
    to the first </think> is reasoning."""
    think_parts = []
    s = raw
    if started_in_think:
        close = s.find("</think>")
        if close == -1:  # still inside the opening reasoning block
            return "", (s if final else _safe_cut(s))
        think_parts.append(s[:close])
        s = s[close + len("</think>"):]
    # `.append(...) or ""` is intentional: append returns None, so the replacement is
    # always "" while capturing the matched group as a side effect.
    s = re.sub(r"<think>(.*?)</think>",
               lambda m: think_parts.append(m.group(1)) or "",  # type: ignore[func-returns-value]
               s, flags=re.DOTALL)
    if "<think>" in s:  # reasoning still open: everything after the tag is current thought
        pre, _, post = s.partition("<think>")
        think_parts.append(post)
        s = pre
    think = "".join(think_parts)

    s = re.sub(r"<tool_call>.*?</tool_call>", "", s, flags=re.DOTALL)
    s = re.sub(r"<function=.*?</function>", "", s, flags=re.DOTALL)
    for opener in _TOOL_OPENERS:  # a tool call still being written: hide from its start
        i = s.find(opener)
        if i != -1:
            s = s[:i]
    prose = s
    if not final:
        prose, think = _safe_cut(prose), _safe_cut(think)
    return prose, think


class _StreamView:
    """Feeds raw tokens in, emits 'stream' (prose) and 'think' (reasoning) deltas."""

    def __init__(self, emit, started_in_think: bool = False):
        self._emit = emit
        self._started_in_think = started_in_think
        self.raw = ""
        self._prose = 0   # chars of prose already emitted
        self._think = 0   # chars of reasoning already emitted

    def feed(self, t: str):
        self.raw += t
        self._update(final=False)

    def close(self):
        self._update(final=True)

    @property
    def saw_prose(self) -> bool:
        return self._prose > 0

    def _update(self, final: bool):
        prose, think = _split_view(self.raw, final, self._started_in_think)
        if len(think) > self._think:
            self._emit("think", think[self._think:])
            self._think = len(think)
        if len(prose) > self._prose:
            self._emit("stream", prose[self._prose:])
            self._prose = len(prose)


# ---------------------------------------------------------------------------
# Tool activity rendering (Claude-style): a verb + target on one line, then a
# compact summary — line counts for reads, a +/- diff for edits, a few lines of
# output for bash. Never the whole file. start() is emitted before the tool runs
# (so the spinner can show the right verb during a slow command); result() after.
# ---------------------------------------------------------------------------

_VERB = {"read": "Read", "edit": "Edit", "write": "Write", "bash": "Run",
         "grep": "Search", "glob": "Find", "write_todos": "Plan",
         "repo_map": "Mapping", "task": "Task"}


def _disp_path(p) -> str:
    try:
        r = os.path.relpath(str(p))
        return r if not r.startswith("../../") else str(p)
    except (ValueError, TypeError):
        return str(p)


def _oneline(s: str, n: int = 72) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _firstline(s: str) -> str:
    s = str(s).strip().strip("[]").strip()
    return s.splitlines()[0] if s else ""


def _nlines(s: str) -> int:
    s = str(s)
    return 0 if not s else s.count("\n") + (0 if s.endswith("\n") else 1)


def _is_err(result: str) -> bool:
    # Display-only heuristic: does this tool result read as a chad error message? Those
    # are single bracketed diagnostics (`[no such file: …]`, `[exit 1]\n…`), so require a
    # leading `[` AND scan only the FIRST line — otherwise a `read`/grep
    # of `[`-leading multi-line content (a JSON array, a TOML/markdown doc) would smuggle a
    # keyword in from a later line and get mis-styled as an error.
    r = str(result)
    if not r.startswith("["):
        return False
    head = r.split("\n", 1)[0][:48].lower()
    return any(k in head for k in (
        "no such file", "cannot read", "error", "not found", "denied", "bad regex",
        "timed out", "exit ", "must be", "missing", "unknown tool", "no-op", "appears",
        "old string", "empty"))


def _indent_block(emit, text: str, kind: str = "muted", max_lines: int = 6):
    text = str(text).rstrip("\n")
    if not text:
        emit("muted", "  ⎿ (no output)")
        return
    lines = text.split("\n")
    for i, ln in enumerate(lines[:max_lines]):
        emit(kind, ("  ⎿ " if i == 0 else "     ") + ln)
    if len(lines) > max_lines:
        emit("muted", f"     … +{len(lines) - max_lines} lines")


def _emit_diff(emit, old: str, new: str, max_lines: int = 30, filename: str = ""):
    diff = [d for d in difflib.unified_diff(
        str(old).splitlines(), str(new).splitlines(), lineterm="", n=2)
        if not d.startswith(("---", "+++", "@@"))]
    adds = sum(1 for d in diff if d.startswith("+"))
    dels = sum(1 for d in diff if d.startswith("-"))
    emit("muted", f"  ⎿ +{adds} -{dels}")
    for d in diff[:max_lines]:
        # Syntax colors live *inside* the +/- line color (the +/- kind is the outer
        # layer). `_highlight_code` is a no-op without pygments, so the plain path is
        # byte-identical to the pre-042 output.
        if d.startswith("+"):
            emit("add", "  + " + _highlight_code(d[1:], filename))
        elif d.startswith("-"):
            emit("del", "  - " + _highlight_code(d[1:], filename))
        else:
            emit("muted", "    " + _highlight_code(d[1:], filename))
    if len(diff) > max_lines:
        emit("muted", f"     … +{len(diff) - max_lines} more diff lines")


def render_tool_start(emit, name: str, args: dict):
    """One-line header shown before the tool executes."""
    if name == "read":
        emit("tool", f"Read {_disp_path(args.get('path', ''))}")
    elif name in ("edit", "write"):
        emit("tool", f"{_VERB[name]} {_disp_path(args.get('path', ''))}")
    elif name == "bash":
        emit("tool", f"Run  {_oneline(args.get('command', ''))}")
    elif name == "grep":
        emit("tool", f"Search {_oneline(args.get('pattern', ''), 50)!r}")
    elif name == "glob":
        emit("tool", f"Find {_oneline(args.get('pattern', ''), 50)}")
    elif name == "write_todos":
        emit("tool", "Plan")
    elif name == "task":
        emit("tool", f"Task {_oneline(args.get('description', ''), 50)}")
    elif name == "overview":
        emit("tool", f"Overview {_disp_path(args.get('path', ''))}")
    elif name == "view_symbol":
        emit("tool", f"View {args.get('name', '')}")
    elif name == "find_symbol":
        emit("tool", f"Find {args.get('name', '')}")
    elif name == "find_refs":
        emit("tool", f"Refs {args.get('name', '')}")
    elif name in ("replace_symbol", "insert_symbol"):
        emit("tool", f"Edit {args.get('name', '')}")
    elif name == "rename_symbol":
        emit("tool", f"Rename {args.get('name', '')} → {args.get('new_name', '')}")
    elif name.startswith("mcp__"):
        # mcp__<server>__<tool> -> "MCP server/tool {args…}" so a trace reads cleanly.
        bits = name[len("mcp__"):].split("__", 1)
        loc = " / ".join(bits) if len(bits) == 2 else name
        emit("tool", f"MCP {loc} {_oneline(_compact_args(args), 60)}")
    else:
        emit("tool", name)


def render_tool_result(emit, name: str, args: dict, result: str):
    """Compact summary shown after the tool returns."""
    result = str(result)
    # Symbolic edits: result is "[summary]\n<unified diff>" on success.
    if name in ("replace_symbol", "insert_symbol"):
        if result.startswith(("[replaced", "[inserted")):
            head, _, body = result.partition("\n")
            emit("muted", "  ⎿ " + _firstline(head))
            for d in body.split("\n")[:40]:
                if d.startswith("+"):
                    emit("add", "  + " + d[1:])
                elif d.startswith("-"):
                    emit("del", "  - " + d[1:])
                elif d:
                    emit("muted", "    " + d)
        else:
            emit("error", "  ⎿ " + _firstline(result))
        return
    if name == "rename_symbol":
        emit("muted" if result.startswith("[renamed") else "error",
             "  ⎿ " + _firstline(result))
        return
    if _is_err(result):
        emit("error", "  ⎿ " + _firstline(result))
        return
    if name == "read":
        n = _nlines(result)
        emit("muted", f"  ⎿ {n} line{'s' * (n != 1)}")
    elif name == "view_symbol":
        n = _nlines(result)
        emit("muted", f"  ⎿ {n} line{'s' * (n != 1)}")
    elif name == "overview":
        n = _nlines(result)
        emit("muted", f"  ⎿ {n} symbol{'s' * (n != 1)}")
    elif name == "find_symbol":
        n = _nlines(result)
        emit("muted", f"  ⎿ {n} definition{'s' * (n != 1)}")
    elif name == "find_refs":
        n = _nlines(result)
        emit("muted", f"  ⎿ {n} reference{'s' * (n != 1)}")
    elif name == "edit":
        _emit_diff(emit, args.get("old", ""), args.get("new", ""),
                   filename=str(args.get("path", "")))
    elif name == "write":
        content = args.get("content", "")
        emit("muted", f"  ⎿ {_nlines(content)} lines written")
        _emit_diff(emit, "", content, max_lines=12, filename=str(args.get("path", "")))
    elif name == "bash":
        _indent_block(emit, result)
    elif name == "task":
        n = _nlines(result)
        emit("muted", f"  ⎿ sub-agent returned {n} line{'s' * (n != 1)}")
        _indent_block(emit, result, max_lines=4)
    elif name == "grep":
        # `[`-leading lines are notices ("[results truncated: …]"), not path:line: hits.
        lines = [l for l in result.splitlines() if ":" in l and not l.startswith("[")]
        files = len({l.split(":", 1)[0] for l in lines})
        n = len(lines)
        emit("muted", f"  ⎿ {n} match{'es' * (n != 1)} in {files} file{'s' * (files != 1)}")
    elif name == "glob":
        n = 0 if result == "[no matches]" else _nlines(result)
        emit("muted", f"  ⎿ {n} file{'s' * (n != 1)}")
    elif name == "write_todos":
        for line in result.splitlines()[1:]:  # drop the "Plan updated:" header
            emit("muted", "  " + line.strip())
        # Also feed the structured list to the TUI's pinned todo panel.
        # The plain REPL / one-shot emitter drops the `todos` kind (like ctx/gen/prefill),
        # so the inline muted lines above remain its only rendering.
        todos = args.get("todos")
        if isinstance(todos, list):
            emit("todos", json.dumps(todos))
    else:
        _indent_block(emit, result)


def ansi_fragment(kind: str, text: str) -> str | None:
    """The transcript ANSI fragment shared by the REPL emitter and the TUI. Returns None
    for kinds a caller renders specially (stream, user) or drops (gauges/unknowns)."""
    if kind == "think":
        return C_DIM + text + C_RST
    if kind == "tool":
        return f"\n{C_GREEN}●{C_RST} {C_BOLD}{text}{C_RST}\n"
    if kind == "add":
        return f"{C_GREEN}{text}{C_RST}\n"
    if kind == "del":
        return f"{C_RED}{text}{C_RST}\n"
    if kind == "error":
        return f"{C_YEL}{text}{C_RST}\n"
    if kind in ("info", "muted"):
        return f"{C_DIM}{text}{C_RST}\n"
    return None


# chad's moai (🗿) mascot rendered as a bone-club silhouette — the startup banner art.
# Three rows so it sits beside three info lines (name/version, model, cwd), Claude-Code
# style. Kept as a module constant so tests can assert on it without a live engine.
_BANNER_ART = ("▟█▙▂▂▂", "▜█▛▔▔▔", "▘ ▝   ")


def _tilde(path: str) -> str:
    """Collapse the home-directory prefix to ~, like a shell prompt."""
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def banner(model: str, ctx_limit: int | None, mode: str = "normal",
           version: str | None = None, cwd: str | None = None) -> str:
    """The startup banner: bone-club art on the left, live session info on the right.

    Mirrors Claude Code's header — name+version, model+context, and the working
    directory — so a fresh session states what it is at a glance. Returns a plain
    multi-line ANSI string (no trailing newline); the caller emits it verbatim."""
    if version is None:
        from . import __version__ as version
    if cwd is None:
        cwd = os.getcwd()
    ctx = f"{ctx_limit / 1000:.0f}k context" if ctx_limit else "context tbd"
    info = [
        f"{C_BOLD}chad{C_RST} {C_DIM}v{version}{C_RST}",
        f"{model} {C_DIM}· {ctx} · {mode} mode{C_RST}",
        f"{C_DIM}{_tilde(cwd)}{C_RST}",
    ]
    width = max(len(a) for a in _BANNER_ART)
    rows = [f"{C_YEL}{art:<{width}}{C_RST}   {text}"
            for art, text in zip(_BANNER_ART, info)]
    return "\n".join(rows)


def _default_emit(kind: str, text: str):
    """Default emitter: colored stdout, used by the plain REPL and one-shot mode."""
    w = sys.stdout.write
    if kind == "stream":  # DIVERGES from the TUI: green-wrapped here, raw there.
        w(C_GREEN + text + C_RST)
    elif kind == "user":  # DIVERGES from the TUI: single-line here, multi-line there.
        w(f"\n{C_YEL}» {text}{C_RST}\n")
    else:
        frag = ansi_fragment(kind, text)
        if frag is not None:
            w(frag)
        # 'stat', the live-gauge kinds (ctx/gen/prefill/status), the pinned 'todos' panel
        # feed, and any unknown kinds return None and are intentionally dropped from
        # stdout — they belong to the TUI's pinned region, not the plain REPL scrollback.
    sys.stdout.flush()
