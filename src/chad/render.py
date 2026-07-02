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
    head = str(result)[:48].lower()
    return str(result).startswith("[") and any(k in head for k in (
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


def _emit_diff(emit, old: str, new: str, max_lines: int = 30):
    diff = [d for d in difflib.unified_diff(
        str(old).splitlines(), str(new).splitlines(), lineterm="", n=2)
        if not d.startswith(("---", "+++", "@@"))]
    adds = sum(1 for d in diff if d.startswith("+"))
    dels = sum(1 for d in diff if d.startswith("-"))
    emit("muted", f"  ⎿ +{adds} -{dels}")
    for d in diff[:max_lines]:
        if d.startswith("+"):
            emit("add", "  + " + d[1:])
        elif d.startswith("-"):
            emit("del", "  - " + d[1:])
        else:
            emit("muted", "    " + d[1:])
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
        _emit_diff(emit, args.get("old", ""), args.get("new", ""))
    elif name == "write":
        content = args.get("content", "")
        emit("muted", f"  ⎿ {_nlines(content)} lines written")
        _emit_diff(emit, "", content, max_lines=12)
    elif name == "bash":
        _indent_block(emit, result)
    elif name == "task":
        n = _nlines(result)
        emit("muted", f"  ⎿ sub-agent returned {n} line{'s' * (n != 1)}")
        _indent_block(emit, result, max_lines=4)
    elif name == "grep":
        lines = [l for l in result.splitlines() if ":" in l]
        files = len({l.split(":", 1)[0] for l in lines})
        n = len(lines)
        emit("muted", f"  ⎿ {n} match{'es' * (n != 1)} in {files} file{'s' * (files != 1)}")
    elif name == "glob":
        n = _nlines(result)
        emit("muted", f"  ⎿ {n} file{'s' * (n != 1)}")
    elif name == "write_todos":
        for line in result.splitlines()[1:]:  # drop the "Plan updated:" header
            emit("muted", "  " + line.strip())
    else:
        _indent_block(emit, result)


def _default_emit(kind: str, text: str):
    """Default emitter: colored stdout, used by the plain REPL and one-shot mode."""
    w = sys.stdout.write
    if kind == "stream":
        w(C_GREEN + text + C_RST)
    elif kind == "think":
        w(C_DIM + text + C_RST)
    elif kind == "tool":
        w(f"\n{C_GREEN}●{C_RST} {C_BOLD}{text}{C_RST}\n")
    elif kind == "add":
        w(f"{C_GREEN}{text}{C_RST}\n")
    elif kind == "del":
        w(f"{C_RED}{text}{C_RST}\n")
    elif kind == "error":
        w(f"{C_YEL}{text}{C_RST}\n")
    elif kind == "user":
        w(f"\n{C_YEL}» {text}{C_RST}\n")
    elif kind in ("info", "muted"):
        w(f"{C_DIM}{text}{C_RST}\n")
    # 'stat' and unknown kinds are intentionally dropped from stdout.
    sys.stdout.flush()
