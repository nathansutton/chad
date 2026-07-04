"""Tier-1 (no model) tests for the live status line (plans 033/034).

Covers the pure display helpers (`_kfmt`, `_phase_glyph`) and the contract that the
live-readout emit kinds — `gen` (↓ generated), `prefill` (↑ progress), `ctx` (gauge) —
update state but NEVER queue a transcript fragment (they must stay in the pinned status
region, not leak into the terminal scrollback). `_emit` is exercised on a bare instance
so we don't have to construct an Engine/Agent.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad.engine import Engine  # noqa: E402
from chad.tui import TUI, _kfmt, _phase_glyph, _todo_panel_rows  # noqa: E402


def test_kfmt():
    assert _kfmt(900) == "900"
    assert _kfmt(0) == "0"
    assert _kfmt(999) == "999"
    assert _kfmt(8200) == "8.2k"
    assert _kfmt(1000) == "1.0k"
    assert _kfmt(34000) == "34.0k"


def test_phase_glyph():
    # Known verbs map to their state glyph; prefill/compaction labels share the ⋯ glyph;
    # anything unknown falls back to the neutral dot. All one cell wide.
    assert _phase_glyph("Thinking") == "✶"
    assert _phase_glyph("Editing") == "✎"
    assert _phase_glyph("Running") == "▸"
    assert _phase_glyph("Searching") == "⌕"
    assert _phase_glyph("Re-prefilling after compaction") == "⋯"
    assert _phase_glyph("Prefilling context") == "⋯"
    assert _phase_glyph("Compacting context") == "⋯"
    assert _phase_glyph("Whatever") == "•"
    assert all(len(_phase_glyph(p)) == 1
               for p in ("Thinking", "Editing", "Prefilling context", "Whatever"))


def _bare_tui():
    """A TUI with only the fields `_emit` touches — no engine/agent constructed."""
    t = object.__new__(TUI)
    t._pending = []
    t._todos = []
    t._lock = threading.Lock()
    t._phase = "Thinking"
    t._cur_prompt_tokens = 0
    t._gen_tokens = 0
    t._prefilled = 0
    t._prefill_total = 0
    t._think_capped = 0
    return t


def test_live_emit_kinds_queue_no_transcript():
    t = _bare_tui()
    t._emit("gen", "42")
    t._emit("ctx", "100")
    t._emit("prefill", "5/10")
    # State updated...
    assert t._gen_tokens == 42
    assert t._cur_prompt_tokens == 100
    assert (t._prefilled, t._prefill_total) == (5, 10)
    # ...but nothing leaked into the scrollback buffer.
    assert t._pending == []


def test_status_emit_sets_phase_without_transcript():
    t = _bare_tui()
    t._emit("status", "Re-prefilling after compaction")
    assert t._phase == "Re-prefilling after compaction"
    assert t._pending == []


def test_malformed_live_emits_are_ignored():
    t = _bare_tui()
    t._emit("gen", "not-an-int")
    t._emit("prefill", "garbage")
    assert t._gen_tokens == 0
    assert (t._prefilled, t._prefill_total) == (0, 0)
    assert t._pending == []


def test_settle_ctx_gauge_avoids_retokenizing():
    # Plan 044 item 3: at end of turn the ctx gauge is refreshed from the last `ctx`
    # emit (final step's prompt size) + the generated-token count — NOT by re-rendering
    # the whole transcript. The helper is pure arithmetic over already-tracked state; if
    # it ever calls self.agent._render() this bare instance (no .agent) would AttributeError.
    t = _bare_tui()
    t._cur_prompt_tokens = 8000   # last `ctx` emit during the turn
    t._gen_tokens = 240           # tokens the final step generated
    t._settle_ctx_gauge()
    assert t._cur_prompt_tokens == 8240


def test_info_emit_does_queue_a_fragment():
    # Contrast: a real transcript kind still appends (proves the no-queue checks above
    # aren't just because _emit never queues anything).
    t = _bare_tui()
    t._emit("info", "hello")
    assert t._pending and "hello" in "".join(t._pending)


def test_todos_emit_updates_panel_not_scrollback():
    # Plan 042 item 1: the `todos` kind feeds the pinned panel (parsed JSON list) and,
    # like ctx/gen/prefill, never queues a transcript fragment into scrollback.
    t = _bare_tui()
    t._emit("todos", '[{"content": "a", "status": "completed"}, '
                     '{"content": "b", "status": "in_progress"}]')
    assert [x.get("status") for x in t._todos] == ["completed", "in_progress"]
    assert t._pending == []
    # Malformed payload clears rather than raising.
    t._emit("todos", "not json")
    assert t._todos == [] and t._pending == []


def test_thinkcap_emit_updates_state_not_scrollback():
    # Plan 057: the `thinkcap` kind carries the per-turn soft think-cap count for the
    # status line. Like gen/prefill/ctx it updates state but never queues a transcript
    # fragment, and a malformed payload is ignored rather than raising.
    t = _bare_tui()
    t._emit("thinkcap", "2")
    assert t._think_capped == 2
    assert t._pending == []
    t._emit("thinkcap", "not-an-int")
    assert t._think_capped == 2  # unchanged
    assert t._pending == []


def test_thinkcap_glyph_absent_by_default_present_when_capped():
    # Plan 057: the ✂N indicator appears in the busy status line ONLY when the soft
    # think-cap fired this turn (_think_capped > 0). With the cap off (the default,
    # _think_capped == 0) the line is byte-identical to before — no ✂ anywhere.
    tui = TUI(_fake_engine(), ctx_limit=24000)
    tui._busy = True
    tui._turn_start = time.monotonic()

    def line():
        return "".join(txt for _style, txt in tui._status_fragments())

    tui._think_capped = 0
    assert "✂" not in line()

    tui._think_capped = 3
    assert "✂3" in line()


def test_todo_panel_rows_collapse_and_glyphs():
    assert _todo_panel_rows([]) == []
    short = [{"content": "a", "status": "completed"},
             {"content": "b", "status": "in_progress"},
             {"content": "c", "status": "pending"}]
    assert _todo_panel_rows(short) == [
        ("completed", "✓ a"), ("in_progress", "▸ b"), ("pending", "· c")]
    # A long list collapses to one summary row naming the current (in_progress) item.
    long = [{"content": f"t{i}", "status": "completed"} for i in range(4)]
    long += [{"content": "now", "status": "in_progress"}]
    long += [{"content": f"p{i}", "status": "pending"} for i in range(6)]
    rows = _todo_panel_rows(long, max_items=8)
    assert len(rows) == 1 and rows[0][0] == "summary"
    assert rows[0][1].startswith("4/11 done") and "now" in rows[0][1]


# ---------------------------------------------------------------------------
# Worker-thread concurrency (plan 054): queue order, interrupt delivery, the
# cross-thread y/n confirm rendezvous.
#
# `tui.py` runs the agent loop on a background thread pulling from `_queue`, with
# `threading.Event`s for interrupt and confirm. None of that was covered — only the
# pure formatters above were. These pin the concurrency contract WITHOUT a model, a
# network, or a real prompt_toolkit App: the TUI is fully constructed (so the real
# `Agent` is built with `should_stop=self._interrupt.is_set`), then `self.agent` is
# swapped for a controllable stub and the worker/confirm methods are driven directly.
#
# Every coordination point is an Event (or a bounded spin whose *delay* is Event.wait,
# never time.sleep). Threads are joined with a small timeout and asserted finished, so a
# broken wiring FAILS the test (via the join-timeout assert) instead of hanging CI.
# ---------------------------------------------------------------------------

_JOIN = 2.0  # thread-join timeout; a hung worker/confirm trips this instead of hanging


def _fake_engine():
    """An Engine with only the fields TUI/Agent construction touches — no weights."""
    eng = object.__new__(Engine)
    eng.model_id = "test/model"
    eng.kv_bytes_per_token = 20_000.0
    return eng


class _FakeAgent:
    """Drop-in for `TUI.agent`. Records each `run_turn` message and can run a caller-
    supplied hook *inside* the turn (to block, or to observe the interrupt). `save`,
    `budget_note`, `mode`, and `last_plan_path` are the only other members the worker
    reads. `_should_stop` is the SAME callback the real Agent was built with, so an
    interrupt set on the TUI is visible here — this proves the wiring, not a copy of it."""

    def __init__(self, should_stop):
        self._should_stop = should_stop
        self.calls = []
        self.saved = 0
        self.mode = "normal"
        self.budget_note = None
        self.last_plan_path = None
        self.on_call = None  # optional callable(msg) executed within run_turn

    def run_turn(self, msg, stream=True):
        self.calls.append(msg)
        if self.on_call is not None:
            self.on_call(msg)

    def save(self):
        self.saved += 1


def _worker_tui():
    """A fully-constructed TUI (fake engine, no real terminal) with its agent replaced by
    a controllable stub. Returns (tui, fake_agent). `app.invalidate` is stubbed to a
    no-op so `_confirm` doesn't need a running prompt_toolkit event loop."""
    tui = TUI(_fake_engine(), ctx_limit=24000)
    fake = _FakeAgent(tui.agent._should_stop)  # capture the real wiring before swapping
    tui.agent = fake
    tui.app.invalidate = lambda: None
    return tui, fake


def _spin_until(pred, timeout=_JOIN):
    """Wait until `pred()` is true. The poll delay is a never-set Event's timed wait, so
    no time.sleep is used for synchronization; `time.monotonic` only bounds the deadline."""
    idle = threading.Event()
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        idle.wait(0.005)
    return True


def _start_worker(tui):
    th = threading.Thread(target=tui._worker, daemon=True)
    th.start()
    return th


def _stop_worker(tui, th):
    tui._shutdown = True
    tui._wake.set()
    th.join(timeout=_JOIN)
    assert not th.is_alive(), "worker did not shut down"


# -- Step 1: smoke — an empty worker shuts down cleanly ----------------------

def test_worker_shutdown_joins_cleanly():
    tui, fake = _worker_tui()
    th = _start_worker(tui)
    _stop_worker(tui, th)
    assert fake.calls == []


# -- Step 2: queue ordering / type-ahead -------------------------------------

def test_worker_drains_queue_in_order():
    tui, fake = _worker_tui()
    drained = threading.Event()
    msgs = ["first", "second", "third"]
    fake.on_call = lambda _m: drained.set() if len(fake.calls) == len(msgs) else None
    th = _start_worker(tui)
    for m in msgs:
        tui._queue.append(m)
    tui._wake.set()
    assert drained.wait(_JOIN), "worker did not drain every queued message"
    assert fake.calls == msgs  # FIFO, no reordering
    assert fake.saved == len(msgs)  # each turn persisted
    _stop_worker(tui, th)


# -- Step 3: interrupt delivery ----------------------------------------------

def test_interrupt_stops_turn():
    tui, fake = _worker_tui()
    entered = threading.Event()
    returned = threading.Event()

    def block_until_interrupted(_msg):
        entered.set()
        idle = threading.Event()
        while not fake._should_stop():  # the Event wired into the agent's should_stop
            idle.wait(0.01)
        returned.set()

    fake.on_call = block_until_interrupted
    th = _start_worker(tui)
    tui._queue.append("do work")
    tui._wake.set()

    assert entered.wait(_JOIN), "worker never started the turn"
    assert not returned.is_set(), "turn returned before any interrupt"
    tui._interrupt.set()  # the SAME Event the Agent's should_stop closes over
    assert returned.wait(_JOIN), "setting _interrupt did not stop the turn"
    _spin_until(lambda: not tui._busy)
    assert not tui._busy  # worker idle again
    _stop_worker(tui, th)


# -- Step 4: confirm handshake rendezvous ------------------------------------

def test_confirm_blocks_until_answered():
    tui, _ = _worker_tui()
    out = {}
    th = threading.Thread(
        target=lambda: out.__setitem__("v", tui._confirm("write", {"path": "x"})),
        daemon=True,
    )
    th.start()
    # `_confirm` publishes the request before parking on the wait loop.
    assert _spin_until(lambda: tui._confirm_req is not None), "confirm never parked"
    assert th.is_alive() and "v" not in out  # still blocked, no answer yet
    assert tui._confirm_req == ("write", {"path": "x"})

    tui._confirm_answer = True  # the UI answers "yes"
    tui._confirm_event.set()
    th.join(timeout=_JOIN)
    assert not th.is_alive(), "confirm did not return after being answered"
    assert out["v"] is True
    assert tui._confirm_req is None  # cleared on return


def test_confirm_returns_false_when_already_interrupted():
    tui, _ = _worker_tui()
    tui._interrupt.set()
    # Interrupt already pending -> refuse immediately, no thread/rendezvous needed.
    assert tui._confirm("write", {"path": "x"}) is False
    assert tui._confirm_req is None


def test_confirm_returns_false_on_shutdown_midwait():
    tui, _ = _worker_tui()
    out = {}
    th = threading.Thread(
        target=lambda: out.__setitem__("v", tui._confirm("write", {"path": "x"})),
        daemon=True,
    )
    th.start()
    assert _spin_until(lambda: tui._confirm_req is not None), "confirm never parked"
    # Shutdown with the confirm still parked: the 0.1s wait-timeout lets the loop observe
    # `_shutdown` and bail False (mirrors a quit while a confirm is outstanding).
    tui._shutdown = True
    th.join(timeout=_JOIN)
    assert not th.is_alive(), "confirm did not unblock on shutdown"
    assert out["v"] is False
    assert tui._confirm_req is None
