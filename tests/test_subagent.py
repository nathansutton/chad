"""Tests for the subagent/Task tool + engine cache quarantine (plan 041).

Tier 1 (always runs, no model): pins the mechanisms that must hold regardless of the
model — the engine's one-deep cache push/pop stack (RAM path AND the disk-spill path,
via a faked cache_utils), the depth-1 guard, and the tool-schema gating that makes
`task` visible to the main agent but invisible (plus the read-only restriction) to a
spawned sub-agent.

Tier 2 (model-gated, self-skipping) lives in test_engine.py::test_push_pop_bit_exact —
it loads the small trimmable model and proves that prefill A → push → run B → pop leaves
the main cache generating byte-identically to a never-pushed control. That is the
crown-jewel invariant; this file pins everything around it that needs no weights.

Run: `uv run python -m pytest tests/test_subagent.py -q`
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest  # noqa: E402

from chad import engine as eng_mod  # noqa: E402
from chad.engine import Engine  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


# --- a minimal fake Engine (bypass __init__: no weights) ----------------------

def _fake_engine(**over):
    """An Engine with just the fields push_cache/pop_cache touch, so the stack logic can
    be exercised without loading a model. _reset_cache is stubbed to a sentinel object so
    we can prove the live cache was actually swapped out and back."""
    eng = object.__new__(Engine)
    eng._cache = ["MAIN-CACHE"]
    eng._cached_ids = [1, 2, 3, 4, 5]
    eng._trimmable = False
    eng._pld_hybrid = True
    eng._warm_prefix_ids = [1, 2, 3]
    eng._cache_stack = []
    eng.cache_dir = None
    eng.kv_bytes_per_token = 20_000.0
    eng.model_id = "test/model"
    eng._reset_count = [0]

    def _reset():
        eng._reset_count[0] += 1
        eng._cache = ["FRESH-CACHE"]
        eng._cached_ids = []

    eng._reset_cache = _reset
    for k, v in over.items():
        setattr(eng, k, v)
    return eng


def _no_mx(monkeypatch):
    """Neutralize mx.clear_cache (called by push/pop) so the RAM-path tests need no GPU."""
    monkeypatch.setattr(eng_mod.mx, "clear_cache", lambda: None)


# === Tier 1: engine push/pop (RAM path) =======================================

def test_push_pop_ram_roundtrip(monkeypatch):
    """push_cache stashes the live cache + ids + flags and hands out a fresh cache;
    pop_cache restores every field to exactly what it was. This is the core quarantine
    invariant the subagent relies on (the main session comes back bit-identical)."""
    _no_mx(monkeypatch)
    eng = _fake_engine()
    main_cache = eng._cache

    eng.push_cache()
    check("push resets to a fresh empty cache", eng._cached_ids == [] and eng._cache == ["FRESH-CACHE"],
          f"{eng._cached_ids} {eng._cache}")
    check("push kept a stack frame", len(eng._cache_stack) == 1, len(eng._cache_stack))

    eng.pop_cache()
    check("pop restores the exact main cache object", eng._cache is main_cache, eng._cache)
    check("pop restores cached_ids", eng._cached_ids == [1, 2, 3, 4, 5], eng._cached_ids)
    check("pop restores flags", eng._trimmable is False and eng._pld_hybrid is True,
          f"{eng._trimmable} {eng._pld_hybrid}")
    check("pop restores warm_prefix_ids", eng._warm_prefix_ids == [1, 2, 3], eng._warm_prefix_ids)
    check("stack empty after pop", eng._cache_stack == [], eng._cache_stack)


def test_depth_one_guard(monkeypatch):
    """Depth 1 only — a second push while one is live is a programming error (subagents
    can't nest). It must raise and NOT clobber the outstanding frame."""
    _no_mx(monkeypatch)
    eng = _fake_engine()
    eng.push_cache()
    with pytest.raises(RuntimeError):
        eng.push_cache()
    check("guard left the single frame intact", len(eng._cache_stack) == 1, len(eng._cache_stack))


def test_pop_without_push_raises(monkeypatch):
    _no_mx(monkeypatch)
    eng = _fake_engine()
    with pytest.raises(RuntimeError):
        eng.pop_cache()


def test_ram_path_when_no_cache_dir(monkeypatch):
    """With no cache_dir there is nowhere to spill, so _should_spill is False and the
    cache is held in RAM even if it's large."""
    _no_mx(monkeypatch)
    eng = _fake_engine(cache_dir=None, _cached_ids=list(range(100_000)))
    check("no cache_dir -> never spill", eng._should_spill(eng._cached_ids) is False)


# === Tier 1: engine push/pop (disk-spill path, faked cache_utils) =============

def test_push_pop_spill_roundtrip(monkeypatch, tmp_path):
    """When _should_spill says RAM is tight, push serializes the main cache to a disk
    checkpoint and drops the RAM reference; pop reloads it and removes the file. We fake
    save/load_prompt_cache (no MLX) and force the spill decision, then assert the reload
    restores cached_ids and the spill file is cleaned up."""
    _no_mx(monkeypatch)
    saved = {}

    def _fake_save(path, cache):
        saved["path"] = path
        saved["cache"] = list(cache)
        with open(path, "w") as f:
            f.write("x")

    def _fake_load(path):
        return ["RELOADED-CACHE"]

    monkeypatch.setattr(eng_mod.cache_utils, "save_prompt_cache", _fake_save)
    monkeypatch.setattr(eng_mod.cache_utils, "load_prompt_cache", _fake_load)

    eng = _fake_engine(cache_dir=str(tmp_path))
    monkeypatch.setattr(eng, "_should_spill", lambda ids: True)

    eng.push_cache()
    check("spill wrote a checkpoint", os.path.isfile(saved["path"]), saved.get("path"))
    check("spill dropped the RAM cache reference", eng._cache_stack[0]["cache"] is None)
    check("spill saved the main cache contents", saved["cache"] == ["MAIN-CACHE"], saved["cache"])

    spill_path = saved["path"]
    eng.pop_cache()
    check("pop reloaded from disk", eng._cache == ["RELOADED-CACHE"], eng._cache)
    check("pop restored cached_ids", eng._cached_ids == [1, 2, 3, 4, 5], eng._cached_ids)
    check("pop removed the spill file", not os.path.isfile(spill_path), spill_path)


def test_spill_ckpt_path_namespaced(monkeypatch, tmp_path):
    """The push-spill checkpoint is namespaced (tag='push') so it can never collide with
    a warm-prefix checkpoint that happens to share the same ids."""
    eng = _fake_engine(cache_dir=str(tmp_path))
    warm = eng._ckpt_path([1, 2, 3])
    push = eng._ckpt_path([1, 2, 3], tag="push")
    check("tagged path differs from untagged", warm != push, f"{warm} == {push}")


# === Tier 1: tool-schema gating (task visibility + reentrancy + read-only) =====

def test_task_in_default_schemas():
    from chad.tools import active_schemas
    names = {s["function"]["name"] for s in active_schemas()}
    check("task exposed by default", "task" in names, names)


def test_task_hidden_under_no_task(monkeypatch):
    monkeypatch.setenv("CHAD_NO_TASK", "1")
    from chad.tools import active_schemas
    names = {s["function"]["name"] for s in active_schemas()}
    check("CHAD_NO_TASK hides task", "task" not in names, names)


def test_task_has_a_validation_schema():
    """task is dispatched specially (not via DISPATCH) but must still validate, so its
    parameter schema has to be registered with validate.PARAM_SCHEMAS."""
    from chad.validate import _param_schema
    schema = _param_schema("task")
    check("task param schema registered", schema is not None)
    check("task requires description+prompt",
          set(schema.get("required", [])) == {"description", "prompt"},
          schema.get("required"))


class _NoModelEngine:
    """Stand-in engine for Agent construction: Agent.__init__ never calls the model, it
    only reads engine for later. But _active_schemas needs nothing from it."""
    tok = None
    model_id = "test/model"
    cache_dir = None


def _mk_agent(**kw):
    from chad.agent import Agent
    return Agent(_NoModelEngine(), **kw)


def test_toplevel_agent_sees_task():
    names = {s["function"]["name"] for s in _mk_agent()._active_schemas()}
    check("top-level agent sees task", "task" in names, names)


def test_subagent_never_sees_task():
    """Reentrancy guard: a sub-agent's schema omits task — subagents cannot spawn
    subagents (enforced in _active_schemas, on top of the depth-1 engine guard)."""
    names = {s["function"]["name"] for s in _mk_agent(subagent=True)._active_schemas()}
    check("sub-agent cannot see task", "task" not in names, names)


def test_subagent_readonly_restriction():
    """A read-only sub-agent sees ONLY the read-only exploration set — no bash/write/edit
    or symbol editors, so it can't mutate the repo it's spelunking through."""
    from chad.agent import SUBAGENT_READ_ONLY
    names = {s["function"]["name"] for s in _mk_agent(subagent=True)._active_schemas()}
    check("no mutating tools for read-only sub-agent",
          not (names & {"bash", "write", "edit", "replace_symbol", "insert_symbol", "rename_symbol"}),
          names)
    check("read-only set is a subset of the allowlist", names <= SUBAGENT_READ_ONLY, names)


def test_subagent_all_keeps_mutators_but_not_task():
    """tools='all' restores the mutating tools (the escape hatch) but STILL drops task."""
    names = {s["function"]["name"]
             for s in _mk_agent(subagent=True, subagent_tools="all")._active_schemas()}
    check("tools=all keeps edit/bash", {"bash", "edit", "write"} <= names, names)
    check("tools=all still drops task", "task" not in names, names)


def test_subagent_skips_session_reset(monkeypatch):
    """A top-level Agent resets the skills/MCP session; a sub-agent must NOT (it shares
    the parent's live connections). Verify reset_session is called only for the parent."""
    from chad import mcp, skills
    calls = []
    monkeypatch.setattr(skills, "reset_session", lambda: calls.append("skills"))
    monkeypatch.setattr(mcp, "reset_session", lambda: calls.append("mcp"))
    _mk_agent(subagent=True)
    check("sub-agent skips session reset", calls == [], calls)
    _mk_agent()
    check("top-level agent resets the session", set(calls) == {"skills", "mcp"}, calls)


# === Tier 1: autonomy clamp (plan 048) ========================================

def test_subagent_tools_policy():
    """A sub-agent auto-approves its own tool calls, so it must never hold more autonomy
    than its parent: only an 'auto' (--yolo/headless) parent may delegate 'all'. Anything
    else — including a 'normal' parent whose safety promise is human confirmation of every
    mutation — clamps to read-only."""
    from chad.agent import subagent_tools_for
    for parent_mode, requested, expected in [
        ("auto", "all", "all"),
        ("normal", "all", "read-only"),
        ("plan", "all", "read-only"),
        ("normal", "read-only", "read-only"),
        ("auto", "read-only", "read-only"),
    ]:
        check(f"subagent_tools_for({parent_mode!r}, {requested!r})",
              subagent_tools_for(parent_mode, requested) == expected,
              subagent_tools_for(parent_mode, requested))


def test_normal_parent_never_spawns_mutating_subagent(monkeypatch):
    """Integration seam: _run_subagent on a mode='normal' parent downgrades an explicit
    tools='all' request and says so in the transcript (the muted clamp notice), while a
    default read-only request stays silent. run_turn is stubbed class-level so no model
    or real sub-agent turn is needed."""
    from chad.agent import Agent
    monkeypatch.setattr(Agent, "run_turn", lambda self, prompt, stream=True: "ok")
    agent = _mk_agent(mode="normal")
    agent.engine.push_cache = lambda: None
    agent.engine.pop_cache = lambda: None
    seen = []
    agent._emit = lambda k, t: seen.append((k, t))

    result = agent._run_subagent("d", "p", tools="all")
    check("stubbed sub-agent turn ran", result == "ok", result)
    check("clamp notice emitted for downgraded 'all'",
          any(k == "muted" and "clamped to read-only" in t for k, t in seen), seen)

    seen.clear()
    agent._run_subagent("d", "p", tools="read-only")
    check("no clamp notice for a default read-only request",
          not any("clamped" in t for _, t in seen), seen)


# === Tier 1: dimmed sub-agent emit remapping ==================================

def test_sub_emit_remaps_kinds():
    """_sub_emit passes live gauges through, suppresses prose/reasoning, and downgrades
    tool/notice kinds to the dim 'muted' channel so sub-agent activity reads as
    subordinate in the main transcript."""
    seen = []
    agent = _mk_agent()
    agent._emit = lambda k, t: seen.append((k, t))
    agent._sub_emit("status", "Thinking")
    agent._sub_emit("ctx", "1234")
    agent._sub_emit("stream", "hello prose")
    agent._sub_emit("think", "secret reasoning")
    agent._sub_emit("tool", "Read foo.py")
    agent._sub_emit("info", "some notice")
    kinds = [k for k, _ in seen]
    check("status passes through", ("status", "Thinking") in seen, seen)
    check("ctx passes through", ("ctx", "1234") in seen, seen)
    check("stream + think suppressed", "stream" not in kinds and "think" not in kinds, seen)
    check("tool downgraded to muted", any(k == "muted" and "Read foo.py" in t for k, t in seen), seen)
    check("info downgraded to muted", any(k == "muted" and "some notice" in t for k, t in seen), seen)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    import inspect
    for fn in tests:
        sig = inspect.signature(fn)
        if sig.parameters:  # needs pytest fixtures (monkeypatch/tmp_path)
            print(f"  (skipping {fn.__name__} — needs pytest fixtures; run under pytest)")
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            FAIL += 1
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
