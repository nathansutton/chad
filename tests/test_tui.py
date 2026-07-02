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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

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
