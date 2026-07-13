"""Terminal UI for chad — the part that makes it feel like Claude Code.

Features that close the gap with Claude Code's UX:
  * shift-tab cycles permission modes: normal -> auto-accept edits -> plan mode
  * type-ahead message queue: keep typing while the agent works; messages run in order
  * ctrl-c interrupts the running turn (stops generation) without killing the session
  * inline y/n approval for mutating tools in normal mode
  * a live status line: model, mode, queued count, context usage

Rendering model (this is why copy/paste and scrolling work like a normal terminal):
the transcript is printed straight into the terminal's *normal* scrollback as ANSI
text — there is no alternate screen and no mouse capture. Only the status line and
the input box live in a small prompt_toolkit region pinned at the bottom
(`full_screen=False`), and background output is routed above it via `patch_stdout`.
Because the app never grabs the mouse or switches to the alt screen, the terminal's
own text selection (copy/paste) and scrollback behave exactly as they do at a shell.

Architecture: the agent loop runs on a background worker thread pulling from a
message queue; the prompt_toolkit Application owns the input/status on the asyncio
loop. The agent's I/O is injected (emit/confirm/should_stop callbacks) so the same
Agent code drives both this TUI and the plain REPL. Emitted fragments are buffered
and flushed to stdout on a UI-loop refresher (~20 Hz) so a fast token stream doesn't
trigger a redraw per token.
"""

import asyncio
import json
import os
import re
import sys
import threading
import time
from collections import deque
from typing import Optional

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from . import config
from .agent import INIT_PROMPT, MODE_LABEL, Agent
from .base_engine import BaseEngine
from .ignore import IGNORE_DIRS
from .render import C_RST, C_YEL, ansi_fragment, banner, confirm_preview, render_tool_result

# Styling for the pinned bottom region only (status line + input). The transcript
# above is plain ANSI (see _ansi_for), so it lives in normal terminal scrollback.
_STYLE = Style.from_dict({
    "spinner": "#8fce8f bold",
    "idle": "#6b6b6b",
    "user": "#d7a86e bold",
    "status.normal": "reverse",
    "status.auto": "bg:#3a5f3a #ffffff",
    "status.plan": "bg:#3a3a6f #ffffff",
    "confirm": "bg:#6f5a2a #ffffff",
    "todo.done": "#6b8f6b",
    "todo.cur": "#d7a86e bold",
    "todo.pending": "#8a8a8a",
    "todo.summary": "#8a8a8a",
})

MODE_STYLE = {"normal": "status.normal", "auto": "status.auto", "plan": "status.plan"}

# Spinner frames + the gerund shown next to it, keyed off the latest activity.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_PHASE_VERB = {"Read": "Reading", "Edit": "Editing", "Write": "Writing", "Run": "Running",
               "Search": "Searching", "Find": "Searching", "Plan": "Planning",
               "Overview": "Reading", "View": "Reading", "Refs": "Searching",
               "Task": "Delegating"}

# A one-cell state glyph coupled to the activity verb, so the indicator carries phase at
# a glance (the spinner braille is the *motion*; this is the *identity*). Matched on a
# verb prefix; prefill/compaction labels and anything unknown fall back to a neutral dot.
_PHASE_GLYPH = {"Thinking": "✶", "Planning": "✶", "Responding": "◆", "Reading": "◇",
                "Editing": "✎", "Writing": "✎", "Running": "▸", "Searching": "⌕"}


def _phase_glyph(phase: str) -> str:
    g = _PHASE_GLYPH.get(phase)
    if g:
        return g
    low = phase.lower()
    if "prefill" in low or "compact" in low:
        return "⋯"
    return "•"


def _kfmt(n: int) -> str:
    """Compact token count for the status line: 8200 → "8.2k", <1000 → "900"."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


# ---------------------------------------------------------------------------
# Pinned todo panel (plan 042 item 1). Pure helpers so the collapse/glyph logic is
# unit-testable without constructing a prompt_toolkit layout; the TUI wraps the rows in
# styled fragments. `write_todos` items are {content, status} dicts.
# ---------------------------------------------------------------------------

_TODO_GLYPH = {"completed": "✓", "in_progress": "▸", "pending": "·"}
_TODO_STYLE = {"completed": "class:todo.done", "in_progress": "class:todo.cur",
               "pending": "class:todo.pending", "summary": "class:todo.summary"}


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _todo_status(t) -> str:
    return t.get("status", "pending") if isinstance(t, dict) else "pending"


def _todo_panel_rows(todos, max_items: int = 8):
    """Rows to pin above the status line: `[(kind, text)]`. Empty list → hidden.
    A long list (> max_items) collapses to one summary row (`3/7 done  ▸ current…`);
    otherwise one glyphed row per item (✓ done / ▸ current / · pending)."""
    if not todos:
        return []
    total = len(todos)
    done = sum(1 for t in todos if _todo_status(t) == "completed")
    if total > max_items:
        cur = next((t.get("content", "") for t in todos
                    if isinstance(t, dict) and _todo_status(t) == "in_progress"), "")
        if not cur:
            cur = next((t.get("content", "") for t in todos
                        if isinstance(t, dict) and _todo_status(t) == "pending"), "")
        label = f"{done}/{total} done"
        if cur:
            label += f"  ▸ {_clip(cur, 56)}"
        return [("summary", label)]
    rows = []
    for t in todos:
        st = _todo_status(t)
        content = t.get("content", "") if isinstance(t, dict) else str(t)
        rows.append((st, f"{_TODO_GLYPH.get(st, '·')} {content}"))
    return rows


# ---------------------------------------------------------------------------
# Input completion (plan 042 item 2). The completer wiring is thin; all the logic
# lives in these pure helpers so it is unit-testable (prompt_toolkit Completers are
# awkward to drive in a test — the completer/menu wiring is verified manually).
# ---------------------------------------------------------------------------

# (command, one-line description) for the `/` menu — mirrors _on_accept + /help.
SLASH_COMMANDS = [
    ("/help", "commands & keybindings"),
    ("/init", "analyze the project, write CLAUDE.md"),
    ("/skills", "list available Agent Skills"),
    ("/mcp", "MCP server status"),
    ("/mcp trust", "trust this project's .mcp.json servers"),
    ("/mcp login", "authenticate an MCP server (OAuth)"),
    ("/compact", "reclaim context now"),
    ("/resume", "list recent sessions; /resume <n> forks one"),
    ("/reset", "clear the conversation + KV cache"),
    ("/clear", "clear the conversation + KV cache"),
    ("/model", "show model + context window"),
    ("/mode", "cycle permission mode"),
    ("/accept", "accept a pending plan and implement it"),
    ("/exit", "quit chad"),
    ("/quit", "quit chad"),
]


def slash_matches(text: str):
    """`[(cmd, desc)]` whose command starts with the typed line. Only fires for a
    single-line input that starts with `/`; once the text runs past a known command
    (an arg is being typed, e.g. `/mcp login foo`) nothing matches, so completion stops."""
    if "\n" in text or not text.startswith("/"):
        return []
    return [(c, d) for (c, d) in SLASH_COMMANDS if c.startswith(text)]


def at_path_token(text_before_cursor: str) -> Optional[str]:
    """The `@`-path fragment under the cursor (text AFTER the `@`), or None when the
    cursor isn't in an `@`-token. The token is the last whitespace-delimited chunk."""
    if not text_before_cursor or "@" not in text_before_cursor:
        return None
    frag = re.split(r"\s", text_before_cursor)[-1]
    if not frag.startswith("@"):
        return None
    return frag[1:]


def path_matches(fragment: str, cwd: Optional[str] = None):
    """Filesystem completions for an `@`-path `fragment` (text after `@`). Directories
    get a trailing `/`; `IGNORE_DIRS` are skipped; dotfiles are hidden until a leading
    dot is typed. Returns sorted display strings (the path, without the leading `@`)."""
    base = cwd or os.getcwd()
    dirname, partial = os.path.split(fragment or "")
    listdir = dirname if os.path.isabs(dirname) else os.path.join(base, dirname)
    try:
        entries = os.listdir(listdir or ".")
    except OSError:
        return []
    out = []
    for name in entries:
        if name in IGNORE_DIRS or not name.startswith(partial):
            continue
        if not partial and name.startswith("."):
            continue
        disp = os.path.join(dirname, name) if dirname else name
        if os.path.isdir(os.path.join(listdir, name)):
            disp += "/"
        out.append(disp)
    return sorted(out)


class _ChadCompleter(Completer):
    """Slash-command menu at line start + `@`-path filesystem completion. Thin wrapper
    over the pure helpers above; keeps ⏎-submits/multiline editing untouched."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        for cmd, desc in slash_matches(text):
            yield Completion(cmd, start_position=-len(text), display=cmd, display_meta=desc)
        if text.startswith("/"):
            return
        frag = at_path_token(text)
        if frag is not None:
            for disp in path_matches(frag):
                yield Completion(disp, start_position=-len(frag), display=disp)


def _make_history():
    """Persistent input history: `FileHistory` at ~/.chad/history (mode 0600 — it can
    hold typed paths/snippets, like the session store), or `InMemoryHistory` when
    CHAD_NO_SESSION_LOG is set. Falls back to in-memory on any filesystem error."""
    if config.flag("CHAD_NO_SESSION_LOG"):
        return InMemoryHistory()
    path = os.path.expanduser("~/.chad/history")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))
        else:
            os.chmod(path, 0o600)
        return FileHistory(path)
    except OSError:
        return InMemoryHistory()


class TUI:
    def __init__(self, engine: BaseEngine, ctx_limit: int, mode: str = "normal",
                 thinking: bool = True, max_chars: int = 400_000, resume: list = None,
                 ctx_window: int = None, finalize=None, ctx_limit_fn=None):
        self.engine = engine
        self.ctx_limit = ctx_limit
        self._ctx_limit_fn = ctx_limit_fn  # live per-turn recheck (plan 075 WS1.4)
        self.ctx_window = ctx_window or ctx_limit  # window shown in the banner
        self.thinking = thinking
        self._resume = resume

        # Background model load (plan: instant startup). When `finalize` is given the
        # weights load on a worker thread while the banner + input are already on screen;
        # `_model_ready` gates the turn worker until they're in, `_load_error` records a
        # load failure so a queued turn reports it instead of hanging. With no finalize
        # (tests, or an already-loaded engine) the model is ready immediately.
        self._finalize = finalize
        self._model_ready = threading.Event()
        self._load_error = None
        if finalize is None:
            self._model_ready.set()

        self._pending = []                 # ANSI chunks awaiting flush to stdout
        self._todos = []                   # live write_todos list for the pinned panel
        self._lock = threading.Lock()
        self._queue = deque()              # user messages awaiting the worker
        self._steer_queue = deque()        # mid-run steering awaiting the agent's drain
        self._wake = threading.Event()     # signal the worker that work/queue changed
        self._shutdown = False
        self._busy = False
        self._cur_prompt_tokens = 0        # last rendered prompt size (context gauge)
        self._tick = 0                     # animation frame counter (spinner)
        self._phase = "Thinking"           # current activity verb shown by the spinner
        # Live activity readouts for the bottom status line (plans 033/034). Reset per
        # turn in _worker; updated by the agent's gen/prefill emits. Display-only.
        self._turn_start = 0.0             # time.monotonic() at turn start (elapsed timer)
        self._gen_tokens = 0               # ↓ generated this step (live)
        self._prefilled = 0                # ↑ tokens prefilled so far (live)
        self._prefill_total = 0            # total tokens in the active prefill (for %)
        self._think_capped = 0             # soft think-cap firings this turn (plan 057)

        # interrupt + confirmation plumbing between UI and worker threads
        self._interrupt = threading.Event()
        self._confirm_req: Optional[tuple] = None  # (name, args) awaiting a y/n answer
        self._confirm_event = threading.Event()
        self._confirm_answer = False

        self.agent = Agent(
            engine, ctx_limit=ctx_limit, mode=mode, thinking=thinking,
            emit=self._emit, confirm=self._confirm, should_stop=self._interrupt.is_set,
            drain_steering=self._drain_steering,
            resume=resume, persist=True, ctx_limit_fn=ctx_limit_fn,
        )

        # Plan-mode handoff state. After a plan-mode turn writes a plan file,
        # `_pending_plan` holds its path and the user can steer (type) or accept
        # (ctrl-g / `/accept`). The accepted implementation session inherits the
        # session's baseline permission mode (auto when launched --yolo, else normal).
        self._pending_plan = None
        self._base_mode = self.agent.mode if self.agent.mode != "plan" else "normal"
        # Governor handoff (plan 040): after a turn hard-stops on its budget, holds the
        # deterministic progress note. The next typed message relaunches a FRESH turn
        # (cleared context) seeded with the note — shedding the ramble + huge prefill.
        self._pending_budget_note = None
        # Sessions last shown by a bare `/resume` (so `/resume <n>` maps a number to a
        # session without re-listing / a race with new saves).
        self._resume_list = []

        # Multiline input that auto-grows up to 8 rows. Enter submits; alt-enter
        # (or ctrl-j) inserts a newline; pasted text keeps its newlines.
        self.input = TextArea(
            height=Dimension(min=1, max=8), multiline=True, wrap_lines=True,
            prompt="» ", style="class:user", history=_make_history(),
            accept_handler=self._on_accept,
            completer=_ChadCompleter(), complete_while_typing=True,
        )
        self.status = Window(
            content=FormattedTextControl(self._status_fragments), height=1,
        )
        # A pinned todo checklist above the status line (plan 042 item 1). Hidden when
        # empty; `dont_extend_height` so it takes exactly its row count — no blank rows.
        self.todo_panel = ConditionalContainer(
            Window(content=FormattedTextControl(self._todo_fragments),
                   dont_extend_height=True),
            filter=Condition(lambda: bool(_todo_panel_rows(self._todos))),
        )
        # Only the todo panel + status line + input are owned by prompt_toolkit; the
        # transcript is printed above this region into the terminal's normal scrollback.
        # A FloatContainer hosts the `/` + `@` completion menu (plan 042 item 2).
        root = FloatContainer(
            HSplit([self.todo_panel, self.status, self.input]),
            floats=[Float(xcursor=True, ycursor=True,
                          content=CompletionsMenu(max_height=8, scroll_offset=1))],
        )
        self.app = Application(
            layout=Layout(root, focused_element=self.input),
            key_bindings=self._bindings(),
            style=_STYLE,
            full_screen=False,
            mouse_support=False,
        )

    # -- agent I/O callbacks (called from the worker thread) --------------

    def _emit(self, kind: str, text: str):
        # Map activity to the spinner verb, then queue the transcript fragment (if any).
        if kind == "think":
            self._phase = "Thinking"
        elif kind == "status":  # explicit activity verb from the agent (no transcript)
            self._phase = text
            return
        elif kind == "ctx":     # live prompt-size gauge for the status line
            try:
                self._cur_prompt_tokens = int(text)
            except ValueError:
                pass
            return
        elif kind == "gen":     # live ↓ generated-token count (no transcript)
            try:
                self._gen_tokens = int(text)
            except ValueError:
                pass
            return
        elif kind == "prefill":  # live ↑ prefill progress as "done/total" (no transcript)
            try:
                done, total = text.split("/", 1)
                self._prefilled = int(done)
                self._prefill_total = int(total)
            except (ValueError, IndexError):
                pass
            return
        elif kind == "thinkcap":  # soft think-cap fired this turn (plan 057; no transcript)
            try:
                self._think_capped = int(text)
            except ValueError:
                pass
            return
        elif kind == "todos":   # write_todos payload for the pinned panel (no transcript)
            try:
                self._todos = json.loads(text)
            except (ValueError, TypeError):
                self._todos = []
            return
        elif kind == "stream":
            self._phase = "Responding"
        elif kind == "tool":
            self._phase = _PHASE_VERB.get(text.split(" ", 1)[0], "Working")
        frag = self._ansi_for(kind, text)
        if frag:
            with self._lock:
                self._pending.append(frag)

    def _ansi_for(self, kind: str, text: str) -> str:
        # Transcript fragments as raw ANSI for the terminal scrollback. Prose is
        # left at the terminal's default foreground (near-white); reasoning is dim.
        if kind == "stream":  # DIVERGES from the REPL: raw here, green-wrapped there.
            return text
        if kind == "user":  # DIVERGES from the REPL: multi-line here, single-line there.
            return self._user_ansi(text)
        frag = ansi_fragment(kind, text)
        return frag if frag is not None else ""  # 'stat'/unknowns drop from the UI

    @staticmethod
    def _user_ansi(text: str) -> str:
        # Render a (possibly multiline) user message with the prompt marker on the
        # first line and aligned continuation on the rest.
        lines = text.split("\n")
        out = f"\n{C_YEL}» {lines[0]}{C_RST}\n"
        out += "".join(f"{C_YEL}  {ln}{C_RST}\n" for ln in lines[1:])
        return out

    def _drain_steering(self):
        """Hand queued mid-run steering to the agent (called from the worker thread
        between steps — see agent.py's drain point). deque.popleft is atomic, so the
        UI thread can keep appending while this drains; FIFO order is preserved."""
        out = []
        while self._steer_queue:
            try:
                out.append(self._steer_queue.popleft())
            except IndexError:  # lost a race with another drain — queue is empty
                break
        return out

    def _requeue_leftover_steers(self):
        """A steer with no running turn left to redirect (typed in the turn's last
        moments, during a `!cmd` passthrough, or in the instant between the final
        drain and `_busy` flipping false) must not vanish silently: fall back to
        type-ahead — it runs as the next turn, like before this feature."""
        while self._steer_queue:
            try:
                self._queue.append(self._steer_queue.popleft())
            except IndexError:
                break

    def _confirm(self, name, args) -> bool:
        # Block the worker until the user answers y/n in the UI.
        if self._interrupt.is_set():
            return False
        self._confirm_answer = False
        self._confirm_event.clear()
        self._confirm_req = (name, args)
        self.app.invalidate()
        while not self._confirm_event.wait(timeout=0.1):
            if self._shutdown or self._interrupt.is_set():
                self._confirm_req = None
                return False
        self._confirm_req = None
        return self._confirm_answer

    # -- UI rendering ----------------------------------------------------

    def _status_fragments(self):
        if self._confirm_req:
            name, args = self._confirm_req
            # The status window is a single line (height=1), so flatten the
            # (possibly multi-line) preview into one clipped line.
            preview = confirm_preview(name, args).replace("\n", " ⏎ ")
            if len(preview) > 160:
                preview = preview[:160] + " …"
            return [("class:confirm",
                     f" allow {name}({preview})?  [y]es  [n]o ")]
        mode = self.agent.mode
        pct = int(100 * self._cur_prompt_tokens / self.ctx_limit) if self.ctx_limit else 0
        qn = len(self._queue)
        # Left: an animated activity indicator. Plans 033/034 reverse the old "verb only"
        # choice for THIS live line only (session.log still carries the rate diagnostics):
        # a state glyph + the verb, then the two numbers that are reassurance not noise —
        # elapsed seconds and ↑prefilled/↓generated counts — plus an advancing % while a
        # big prefill streams (the silent gap this is meant to make legible).
        if not self._model_ready.is_set():
            # Startup: weights still loading on the background thread. Show a live spinner
            # so the pinned line reads as "working", not hung, while you type ahead.
            frame = _SPINNER[(self._tick // 2) % len(_SPINNER)]
            label = self.engine.model_id.split("/")[-1]
            hint = "type ahead — runs when ready" if not qn else f"queued:{qn} — runs when ready"
            left = [("class:spinner", f" {frame} loading {label}…  "),
                    ("class:idle", f"{hint} ")]
            return left
        if self._busy:
            frame = _SPINNER[(self._tick // 2) % len(_SPINNER)]
            glyph = _phase_glyph(self._phase)
            elapsed = int(time.monotonic() - self._turn_start) if self._turn_start else 0
            prog = ""
            if self._prefill_total and self._gen_tokens == 0:  # still prefilling this step
                prog = f"{int(100 * self._prefilled / self._prefill_total)}%  "
            # ✂N: the adaptive soft think-cap (plan 039) trimmed N over-long reasoning runs
            # this turn. Shown only when it actually fired — the cap is off by default, so the
            # default line is byte-identical to before (plan 057).
            cap = f"✂{self._think_capped} · " if self._think_capped else ""
            left = [("class:spinner", f" {frame} {glyph} {self._phase}…  "),
                    ("class:idle", f"{prog}{elapsed}s · ↑{_kfmt(self._prefilled)} "
                                   f"↓{_kfmt(self._gen_tokens)} · {cap}ctrl-c ")]
        else:
            left = [("class:idle", f" {_phase_glyph(self._phase)} ready ")]
        # Model id lives in the startup banner now — no need to repeat it every frame.
        bits = [
            f" {MODE_LABEL[mode]} (shift-tab) ",
            f" ctx {pct}% ",
        ]
        if qn:
            bits.append(f" queued:{qn} ")
        sn = len(self._steer_queue)
        if sn:
            bits.append(f" steer:{sn} ")
        return left + [("class:" + MODE_STYLE[mode], "".join(bits))]

    def _todo_fragments(self):
        # prompt_toolkit fragments for the pinned todo panel; one styled row per line.
        frags = []
        for i, (kind, text) in enumerate(_todo_panel_rows(self._todos)):
            if i:
                frags.append(("", "\n"))
            frags.append((_TODO_STYLE.get(kind, "class:todo.pending"), " " + text))
        return frags

    def _flush(self):
        # Runs on the UI loop (refresher). Under patch_stdout this writes the
        # buffered transcript above the pinned input/status region. Coalescing here
        # caps redraws at the refresher rate regardless of token throughput.
        #
        # Critically we do NOT call sys.stdout.flush(): StdoutProxy holds back any
        # text after the last newline, because a partial (newline-less) line would
        # be overwritten by the redraw of the pinned input/status region. Forcing a
        # flush pushes that partial line out and it gets clobbered — which is why an
        # un-flushed stream looked garbled. So we write whole lines as they settle
        # and let the worker emit a trailing newline at turn end (see _worker) to
        # commit the final line.
        with self._lock:
            if not self._pending:
                return
            chunk = "".join(self._pending)
            self._pending.clear()
        sys.stdout.write(chunk)

    async def _refresher(self):
        while not self._shutdown:
            self._flush()
            if self._busy:
                self._tick += 1
            self.app.invalidate()
            await asyncio.sleep(0.05)

    # -- key bindings ----------------------------------------------------

    def _bindings(self):
        kb = KeyBindings()
        confirming = Condition(lambda: self._confirm_req is not None)
        busy = Condition(lambda: self._busy)
        in_input = has_focus(self.input)

        # Enter submits (eager, so it beats the multiline buffer's newline insert);
        # alt-enter and ctrl-j insert a literal newline for multiline prompts.
        @kb.add("enter", filter=in_input & ~confirming, eager=True)
        def _(event):
            self.input.buffer.validate_and_handle()

        @kb.add("escape", "enter", filter=in_input)
        @kb.add("c-j", filter=in_input)
        def _(event):
            self.input.buffer.insert_text("\n")

        @kb.add("s-tab", filter=~confirming)
        def _(event):
            self.agent.cycle_mode()
            event.app.invalidate()

        # ctrl-g accepts a pending plan (clear context + start implementing) when the
        # input is empty; a one-keystroke alternative to typing /accept.
        pending = Condition(lambda: self._pending_plan is not None)
        @kb.add("c-g", filter=in_input & pending)
        def _(event):
            self._accept_plan()
            event.app.invalidate()

        # eager=True so y/n answer the prompt instead of being typed into the input box
        @kb.add("y", filter=confirming, eager=True)
        @kb.add("Y", filter=confirming, eager=True)
        def _(event):
            self._confirm_answer = True
            self._confirm_event.set()

        @kb.add("n", filter=confirming, eager=True)
        @kb.add("N", filter=confirming, eager=True)
        @kb.add("escape", filter=confirming, eager=True)
        def _(event):
            self._confirm_answer = False
            self._confirm_event.set()

        @kb.add("c-c")
        def _(event):
            if self._busy or self._confirm_req:
                self._interrupt.set()
                self._confirm_event.set()  # unblock a pending confirm as a denial
                self._emit("info", "  [interrupting…]")
            elif self.input.text.strip():
                self.input.buffer.reset()
            else:
                self._shutdown_app(event)

        # Escape hard-interrupts a running turn (parity with ctrl-c and Claude Code)
        # so you can stop a trace and steer immediately instead of queueing behind it.
        # Not eager: prompt_toolkit still disambiguates the escape,enter newline combo
        # and meta/alt sequences by the flush timeout. A pending confirm is handled by
        # the eager escape=deny binding above, so this only fires mid-turn.
        @kb.add("escape", filter=busy & ~confirming)
        def _(event):
            self._interrupt.set()
            self._emit("info", "  [interrupting…]")

        @kb.add("c-d")
        def _(event):
            self._shutdown_app(event)

        return kb

    def _shutdown_app(self, event):
        self._shutdown = True
        self._wake.set()
        self._confirm_event.set()
        event.app.exit()

    # -- session reset / plan handoff ------------------------------------

    def _fresh_agent(self, mode: str) -> bool:
        """Clear the conversation + KV cache and start a new Agent in `mode`.
        Returns False (without resetting) if a turn won't yield in time."""
        self._queue.clear()
        self._steer_queue.clear()
        if self._busy:
            # A turn is on the worker thread mutating engine._cache / _cached_ids.
            # Signal it to stop and wait for it to unwind before we reset the cache,
            # otherwise we race the live generate().
            self._interrupt.set()
            deadline = time.time() + 10
            while self._busy and time.time() < deadline:
                time.sleep(0.02)
            if self._busy:
                self._emit("info", "reset deferred: turn still running.")
                return False
        self.agent = Agent(
            self.engine, ctx_limit=self.ctx_limit, mode=mode,
            thinking=self.thinking, emit=self._emit, confirm=self._confirm,
            should_stop=self._interrupt.is_set, drain_steering=self._drain_steering,
            ctx_limit_fn=self._ctx_limit_fn,
        )
        self._pending_plan = None
        self._pending_budget_note = None
        self._interrupt.clear()        # the new turn must start un-interrupted
        self.engine.reset()
        return True

    def _accept_plan(self):
        """Accept a pending plan: clear context and start a fresh implementation
        session (inheriting the session's baseline perms) seeded to execute it."""
        path = self._pending_plan
        if not path:
            self._emit("info", "no plan pending.")
            return
        rel = os.path.relpath(path)
        if not self._fresh_agent(self._base_mode):
            return
        kickoff = (f"Implement the plan in {rel}. Read the whole file first, then "
                   f"execute each step. Run the verification commands when done.")
        self._queue.append(kickoff)
        self._emit("info", f"context cleared · implementing {rel}")
        self._emit("user", kickoff)
        self._wake.set()

    def _handle_resume(self, arg: str):
        """`/resume` lists this directory's recent sessions; `/resume <n>` forks the
        picked one — a fresh Agent (new session_id, cache reset) seeded with the old
        messages, so the original session file is never overwritten (plan 043)."""
        from . import session
        if not arg:
            items = session.list_sessions(os.getcwd(), limit=10)
            if not items:
                self._emit("info", "no saved sessions for this directory.")
                return
            self._resume_list = items
            self._emit("info", "resume which session? type /resume <n>")
            for i, it in enumerate(items, 1):
                self._emit("info", f"  {i}. {session.describe(it)}")
            return
        try:
            n = int(arg)
        except ValueError:
            self._emit("info", "usage: /resume  (to list)  ·  /resume <number>")
            return
        items = self._resume_list or session.list_sessions(os.getcwd(), limit=10)
        if not (1 <= n <= len(items)):
            self._emit("info", "out of range — run /resume to see the list.")
            return
        pick = items[n - 1]
        data = session.load_session(os.getcwd(), pick["session_id"])
        if not data:
            self._emit("info", "could not load that session.")
            return
        if self._busy:
            self._emit("info", "busy — /resume once the current turn finishes.")
            return
        if not self._fresh_agent(self._base_mode):
            return
        # Seed the fresh Agent (which already minted a new session_id) with the restored
        # transcript; next save() writes a NEW file, leaving the picked one untouched.
        self.agent.messages += [m for m in data["messages"] if m.get("role") != "system"]
        self._resume_list = []
        self._emit("info", f"resumed (forked): {session.describe(pick)}")

    # -- input handling --------------------------------------------------

    def _on_accept(self, buff):
        text = buff.text.strip()
        if not text:
            return False
        if text in ("/exit", "/quit"):
            self._shutdown = True
            self._wake.set()
            self.app.exit()
            return False
        # While the weights load in the background the engine isn't built yet, so the
        # commands that reset/compact/reslot the KV cache would crash. Typing a task is
        # fine — it just queues (type-ahead). Everything else waits for the model.
        if not self._model_ready.is_set() and (
                text.startswith(("/reset", "/clear", "/compact", "/resume", "/accept"))):
            self._emit("info", "still loading the model — try that once it's ready.")
            return False
        if text in ("/reset", "/clear"):
            if self._fresh_agent(self.agent.mode):
                self._emit("info", "session reset.")
            return False
        if text == "/accept":
            self._accept_plan()
            return False
        if text == "/mode":
            self.agent.cycle_mode()
            return False
        if text == "/compact":
            # Manual context reclaim. Refuse mid-turn (mutating messages under the
            # worker thread would corrupt the in-flight render); ask the user to wait.
            if self._busy:
                self._emit("info", "busy — /compact again once the current turn finishes.")
            else:
                b, a = self.agent.compact_now()
                self._emit("info", f"compacted context: {b:,}→{a:,} tokens"
                                   + (" (already lean)" if a >= b else ""))
            return False
        if text == "/resume" or text.startswith("/resume "):
            self._handle_resume(text[len("/resume"):].strip())
            return False
        if text == "/model":
            self._emit("info", f"model {self.engine.model_id} · context "
                               f"{self.engine.effective_ctx:,} (compact at {self.ctx_limit:,}) "
                               f"· mode {self.agent.mode}")
            return False
        if text == "/init":
            self._queue.append(INIT_PROMPT)
            self._emit("user", "/init — analyzing the project to write CLAUDE.md"
                       + ("   (queued)" if self._busy else ""))
            self._wake.set()
            return False
        if text == "/skills":
            from . import skills
            for ln in skills.summary_lines():
                self._emit("info", "  " + ln)
            return False
        if text == "/mcp trust":
            from . import mcp
            mcp.trust()
            self._emit("info", "trusted this project — its .mcp.json servers will "
                               "connect on the next turn")
            return False
        if text.startswith("/mcp login"):
            from . import mcp
            name = text[len("/mcp login"):].strip()
            if not name:
                self._emit("info", "usage: /mcp login <server>")
                return False
            self._emit("info", mcp.login(name, emit=lambda m: self._emit("info", m)))
            return False
        if text == "/mcp":
            from . import mcp
            for ln in mcp.summary_lines():
                self._emit("info", "  " + ln)
            return False
        if text == "/help":
            self._emit("info", "shift-tab: cycle mode (normal/auto/plan) · esc/ctrl-c: "
                               "interrupt · /init /skills /mcp /mcp trust /mcp login <server> "
                               "/resume /reset /clear /compact /model /mode /accept /exit · !cmd shell · @path "
                               "attach · type while busy to steer the running turn "
                               "(applies after the current step) · plan ready: type to "
                               "steer, ctrl-g to accept")
            return False
        # A typed message while a governor budget note is pending = continue fresh:
        # clear context and relaunch, seeding the note + the user's steer (plan 040).
        if self._pending_budget_note and not self._busy:
            note = self._pending_budget_note
            self._pending_budget_note = None
            if not self._fresh_agent(self._base_mode):
                return False
            seed = f"{text}\n\n[{note}]"
            self._queue.append(seed)
            self._emit("info", "context cleared · continuing with the progress note")
            self._emit("user", text)
            self._wake.set()
            return False
        # A typed message while a plan is pending = steer: continue the plan-mode
        # session so the model revises the plan file. Drop the pending banner state.
        self._pending_plan = None
        # Mid-run steering (improve 01): input typed while a turn RUNS redirects that
        # turn — the agent injects it after the current step (a pure append, so the
        # warm KV prefix survives) instead of parking it as a new turn that lands too
        # late. `!cmd` keeps the old type-ahead (a shell side-channel, not a message
        # to the model); esc/ctrl-c remain the hard stop.
        if self._busy and not text.startswith("!"):
            self._steer_queue.append(text)
            self._emit("user", text + "   (steering — applies after current step)")
            return False
        # enqueue the message; echo it (note when it's queued behind running work)
        self._queue.append(text)
        self._emit("user", text + ("   (queued)" if self._busy else ""))
        self._wake.set()
        return False

    # -- worker thread ---------------------------------------------------

    def _worker(self):
        while not self._shutdown:
            if not self._queue:
                # Idle sweep: a steer can slip in between the finished turn's final
                # drain and `_busy` flipping false — with no turn left to redirect it
                # would sit in `_steer_queue` forever. Only runs between turns, so it
                # can never steal a steer from a live one.
                self._requeue_leftover_steers()
                if not self._queue:
                    self._wake.wait(timeout=0.2)
                    self._wake.clear()
                    continue
            # Hold queued messages until the background weight load finishes (the input
            # accepts and queues them meanwhile, so typing ahead feels instant).
            if not self._model_ready.is_set():
                self._model_ready.wait(timeout=0.2)
                continue
            if self._load_error:
                self._queue.clear()
                self._emit("error", f"[cannot run — model load failed: {self._load_error}]")
                continue
            msg = self._queue.popleft()
            self._busy = True
            self._turn_start = time.monotonic()  # elapsed timer for the status line
            self._gen_tokens = 0
            self._prefilled = 0
            self._prefill_total = 0
            self._think_capped = 0
            is_shell = msg.startswith("!")
            self._phase = "Running" if is_shell else "Thinking"
            self._interrupt.clear()
            try:
                if is_shell:
                    # `!cmd` shell passthrough (Claude-Code parity): run it directly and
                    # show the output without invoking the model. Interruptible via ctrl-c;
                    # not added to the conversation (it's a side-channel convenience).
                    from .tools import tool_bash
                    cmd = msg[1:].strip()
                    if cmd:
                        self._emit("tool", f"Run  {cmd}")
                        out = tool_bash(cmd, should_stop=self._interrupt.is_set)
                        render_tool_result(self._emit, "bash", {"command": cmd}, out)
                else:
                    self.agent.run_turn(msg, stream=True)
                    self.agent.save()  # persist conversation for --continue
                    # Governor hard-stop (plan 040): the turn ran out of budget with no
                    # landed+verified change. Surface the banked progress note and arm the
                    # fresh-continue handoff — the next typed message starts clean, seeded.
                    if self.agent.budget_note:
                        self._pending_budget_note = self.agent.budget_note
                        self.agent.budget_note = None
                        self._emit("info", "turn hit its budget — no verified change landed.")
                        self._emit("info", "  type to continue fresh (context cleared, "
                                           "seeded with what was learned) — or start a new task")
                    # A finished plan-mode turn that wrote a plan file -> offer the
                    # steer (type) / accept (ctrl-g or /accept) handoff.
                    if self.agent.mode == "plan" and self.agent.last_plan_path:
                        self._pending_plan = self.agent.last_plan_path
                        self.agent.last_plan_path = None
                        rel = os.path.relpath(self._pending_plan)
                        self._emit("info", f"plan ready → {rel}")
                        self._emit("info", "  type to steer · ctrl-g (or /accept) "
                                           "to accept & implement")
            except Exception as e:  # noqa: BLE001 — surface, keep the session alive
                self._emit("error", f"[turn error: {type(e).__name__}: {e}]")
            finally:
                self._requeue_leftover_steers()
                # Commit the turn's trailing (newline-less) line to scrollback and
                # leave a blank line between turns. Without this, the final prose
                # line stays buffered in StdoutProxy until the next newline.
                with self._lock:
                    self._pending.append("\n")
                self._busy = False
                self._settle_ctx_gauge()

    def _settle_ctx_gauge(self):
        """Refresh the end-of-turn context gauge WITHOUT re-tokenizing the transcript
        (plan 044). The turn's last `ctx` emit already set `_cur_prompt_tokens` to the
        final step's prompt size, and generation then appended ~`_gen_tokens` tokens; the
        sum approximates the new context size at ~zero cost. The old
        `len(self.agent._render())` re-ran apply_chat_template over the whole conversation
        on the worker thread just to nudge a display gauge — pure waste on long sessions."""
        self._cur_prompt_tokens += self._gen_tokens

    def _load_model(self):
        """Load the weights off the UI thread, then adopt the RAM-aware compaction limit
        and unblock the turn worker. Runs once, at startup, only when `finalize` was given."""
        try:
            load_s, ctx_limit = self._finalize()
            self.ctx_limit = ctx_limit
            self.agent.ctx_limit = ctx_limit
            self._emit("info", f"ready in {load_s:.0f}s · context {self.engine.effective_ctx:,} "
                               f"(compact at {ctx_limit:,})")
        except Exception as e:  # noqa: BLE001 — surface load failure; don't hang the worker
            self._load_error = f"{type(e).__name__}: {e}"
            self._emit("error", f"[model load failed: {self._load_error}]")
        finally:
            self._model_ready.set()
            self._wake.set()  # nudge the worker if a message was queued while loading

    def _emit_first_task_hint(self):
        """One muted line under the banner on a FRESH session (resume is None): a small
        local model needs a *scoped* ask, so a blinking cursor doesn't invite a
        frontier-sized request that flails. Resumed sessions already have a thread going,
        so they stay hint-free."""
        if self._resume is not None:
            return
        self._emit("muted",
                   'tip: small model, scoped asks — "fix the failing test in '
                   'tests/test_x.py" lands; "improve my codebase" flails. '
                   "shift-tab cycles plan mode.")

    async def run(self):
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        refresher = asyncio.create_task(self._refresher())
        art = banner(self.engine.model_id.split("/")[-1], self.ctx_window,
                     mode=self.agent.mode)
        with self._lock:
            self._pending.append("\n" + art + "\n")
        self._emit("info", "shift-tab for modes · /help")
        self._emit_first_task_hint()
        if self._finalize is not None:
            self._emit("info", f"loading {self.engine.model_id.split('/')[-1]}… "
                               "(type ahead — your first message runs when it's ready)")
            threading.Thread(target=self._load_model, daemon=True).start()
        try:
            # raw=True passes our ANSI through untouched; patch_stdout keeps the
            # input/status region pinned below while output scrolls above it.
            with patch_stdout(raw=True):
                await self.app.run_async()
        finally:
            self._shutdown = True
            self._wake.set()
            refresher.cancel()


def run_tui(engine: BaseEngine, ctx_limit: int, mode: str = "normal", thinking: bool = True,
            resume: list = None, ctx_window: int = None, finalize=None, ctx_limit_fn=None):
    asyncio.run(TUI(engine, ctx_limit, mode=mode, thinking=thinking, resume=resume,
                    ctx_window=ctx_window, finalize=finalize, ctx_limit_fn=ctx_limit_fn).run())
