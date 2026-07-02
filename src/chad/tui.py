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
import os
import sys
import threading
import time
from collections import deque
from typing import Optional

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from .agent import INIT_PROMPT, MODE_LABEL, Agent
from .engine import Engine
from .render import C_BOLD, C_DIM, C_GREEN, C_RED, C_RST, C_YEL, confirm_preview, render_tool_result

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


class TUI:
    def __init__(self, engine: Engine, ctx_limit: int, mode: str = "normal",
                 thinking: bool = True, max_chars: int = 400_000, resume: list = None):
        self.engine = engine
        self.ctx_limit = ctx_limit
        self.thinking = thinking
        self._resume = resume

        self._pending = []                 # ANSI chunks awaiting flush to stdout
        self._lock = threading.Lock()
        self._queue = deque()              # user messages awaiting the worker
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

        # interrupt + confirmation plumbing between UI and worker threads
        self._interrupt = threading.Event()
        self._confirm_req: Optional[tuple] = None  # (name, args) awaiting a y/n answer
        self._confirm_event = threading.Event()
        self._confirm_answer = False

        self.agent = Agent(
            engine, ctx_limit=ctx_limit, mode=mode, thinking=thinking,
            emit=self._emit, confirm=self._confirm, should_stop=self._interrupt.is_set,
            resume=resume, persist=True,
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

        # Multiline input that auto-grows up to 8 rows. Enter submits; alt-enter
        # (or ctrl-j) inserts a newline; pasted text keeps its newlines.
        self.input = TextArea(
            height=Dimension(min=1, max=8), multiline=True, wrap_lines=True,
            prompt="» ", style="class:user", history=InMemoryHistory(),
            accept_handler=self._on_accept,
        )
        self.status = Window(
            content=FormattedTextControl(self._status_fragments), height=1,
        )
        # Only the status line + input are owned by prompt_toolkit; the transcript
        # is printed above this region into the terminal's normal scrollback.
        root = HSplit([self.status, self.input])
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
        if kind == "stream":
            return text
        if kind == "think":
            return C_DIM + text + C_RST
        if kind == "tool":
            return f"\n{C_GREEN}●{C_RST} {C_BOLD}{text}{C_RST}\n"
        if kind == "user":
            return self._user_ansi(text)
        if kind == "add":
            return f"{C_GREEN}{text}{C_RST}\n"
        if kind == "del":
            return f"{C_RED}{text}{C_RST}\n"
        if kind == "error":
            return f"{C_YEL}{text}{C_RST}\n"
        if kind in ("info", "muted"):
            return f"{C_DIM}{text}{C_RST}\n"
        return ""  # 'stat' and unknowns are dropped from the UI

    @staticmethod
    def _user_ansi(text: str) -> str:
        # Render a (possibly multiline) user message with the prompt marker on the
        # first line and aligned continuation on the rest.
        lines = text.split("\n")
        out = f"\n{C_YEL}» {lines[0]}{C_RST}\n"
        out += "".join(f"{C_YEL}  {ln}{C_RST}\n" for ln in lines[1:])
        return out

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
        if self._busy:
            frame = _SPINNER[(self._tick // 2) % len(_SPINNER)]
            glyph = _phase_glyph(self._phase)
            elapsed = int(time.monotonic() - self._turn_start) if self._turn_start else 0
            prog = ""
            if self._prefill_total and self._gen_tokens == 0:  # still prefilling this step
                prog = f"{int(100 * self._prefilled / self._prefill_total)}%  "
            left = [("class:spinner", f" {frame} {glyph} {self._phase}…  "),
                    ("class:idle", f"{prog}{elapsed}s · ↑{_kfmt(self._prefilled)} "
                                   f"↓{_kfmt(self._gen_tokens)} · ctrl-c ")]
        else:
            left = [("class:idle", f" {_phase_glyph(self._phase)} ready ")]
        bits = [
            f" {self.engine.model_id.split('/')[-1]} ",
            f" {MODE_LABEL[mode]} (shift-tab) ",
            f" ctx {pct}% ",
        ]
        if qn:
            bits.append(f" queued:{qn} ")
        return left + [("class:" + MODE_STYLE[mode], "".join(bits))]

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
            should_stop=self._interrupt.is_set,
        )
        self._pending_plan = None
        self._pending_budget_note = None
        self._interrupt.clear()        # the new turn must start un-interrupted
        self.engine._reset_cache()
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
            self._emit("info", "shift-tab: cycle mode (normal/auto/plan) · ctrl-c: "
                               "interrupt · /init /skills /mcp /mcp trust /mcp login <server> "
                               "/reset /clear /compact /model /mode /accept /exit · !cmd shell · @path "
                               "attach · type while busy to queue · plan ready: type to "
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
        # enqueue the message; echo it (note when it's queued behind running work)
        self._queue.append(text)
        self._emit("user", text + ("   (queued)" if self._busy else ""))
        self._wake.set()
        return False

    # -- worker thread ---------------------------------------------------

    def _worker(self):
        while not self._shutdown:
            if not self._queue:
                self._wake.wait(timeout=0.2)
                self._wake.clear()
                continue
            msg = self._queue.popleft()
            self._busy = True
            self._turn_start = time.monotonic()  # elapsed timer for the status line
            self._gen_tokens = 0
            self._prefilled = 0
            self._prefill_total = 0
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
                # Commit the turn's trailing (newline-less) line to scrollback and
                # leave a blank line between turns. Without this, the final prose
                # line stays buffered in StdoutProxy until the next newline.
                with self._lock:
                    self._pending.append("\n")
                self._busy = False
                self._cur_prompt_tokens = len(self.agent._render())

    async def run(self):
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        refresher = asyncio.create_task(self._refresher())
        self._emit("info", f"chad ready · model={self.engine.model_id.split('/')[-1]}"
                           f" · shift-tab for modes · /help")
        try:
            # raw=True passes our ANSI through untouched; patch_stdout keeps the
            # input/status region pinned below while output scrolls above it.
            with patch_stdout(raw=True):
                await self.app.run_async()
        finally:
            self._shutdown = True
            self._wake.set()
            refresher.cancel()


def run_tui(engine: Engine, ctx_limit: int, mode: str = "normal", thinking: bool = True,
            resume: list = None):
    asyncio.run(TUI(engine, ctx_limit, mode=mode, thinking=thinking, resume=resume).run())
