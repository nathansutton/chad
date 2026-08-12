"""Session model v2 — threads, status, prune exemption, cross-project listing, and
the state that must survive a resume (ambient ledger, todos, saved system prompt,
KV-snapshot key plumbing). No model, no network; the engine parts test only the
pre-load path logic (`has_session_snapshot` / path hashing) — the load-bearing
save/load round-trip rides mlx_lm's save/load_prompt_cache, which the warm-prefix
path already exercises on real hardware.

Run: `uv run python -m pytest tests/test_session_v2.py`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_agent_e2e import ScriptedEngine  # noqa: E402

from chad import ambient, session, tools, worktree  # noqa: E402
from chad.agent import Agent  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(worktree, "_ACTIVE", None)
    monkeypatch.setenv("CHAD_WORKTREE_DIR", str(tmp_path / "wt"))
    ambient.reset()
    tools.set_todos([])
    yield
    ambient.reset()
    tools.set_todos([])


# -- ambient + todos round-trip ----------------------------------------------

def test_ambient_snapshot_round_trip():
    ambient.note_call("bash", {"command": "pytest -q"}, "[exit 1]\nboom")
    ambient.note_call("edit", {"path": "a.py"}, "[edited a.py]")
    snap = ambient.snapshot()
    ambient.reset()
    assert ambient.snapshot()["calls"] == 0
    ambient.restore(snap)
    got = ambient.snapshot()
    assert got["calls"] == snap["calls"] == 2
    assert got["edited"] == snap["edited"]
    assert got["last_run"] == snap["last_run"]


def test_ambient_restore_tolerates_garbage():
    ambient.restore({"calls": "not-a-number", "edited": 7})
    assert ambient.snapshot()["calls"] == 0  # fell back to reset, didn't raise


def test_todos_round_trip():
    tools.tool_write_todos([{"content": "step one", "status": "completed"},
                            {"content": "step two", "status": "pending"}])
    saved = tools.current_todos()
    tools.set_todos([])
    assert tools.current_todos() == []
    tools.set_todos(saved)
    assert tools.current_todos() == saved
    tools.set_todos("garbage")  # non-list input clears rather than raises
    assert tools.current_todos() == []


# -- session store v2 ---------------------------------------------------------

def _msgs(text="do the thing"):
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "done"}]


def test_index_rows_carry_thread_status_worktree(tmp_path):
    proj = str(tmp_path / "proj")
    os.makedirs(proj)
    meta = {"thread_id": "T1", "status": "idle",
            "worktree": {"id": "w", "path": "/nope/gone", "branch": "chad/w",
                         "origin_top": proj}}
    session.save_session(proj, _msgs(), meta, session_id="s1")
    rows = session.list_sessions(proj)
    assert rows[0]["thread_id"] == "T1"
    assert rows[0]["status"] == "idle"
    assert rows[0]["worktree_path"] == "/nope/gone"


def test_set_status_updates_file_and_index(tmp_path):
    proj = str(tmp_path / "proj")
    os.makedirs(proj)
    session.save_session(proj, _msgs(), {"status": "idle"}, session_id="s1")
    assert session.set_status(proj, "s1", "applied")
    assert session.load_session(proj, "s1")["meta"]["status"] == "applied"
    assert session.list_sessions(proj)[0]["status"] == "applied"
    assert "applied" in session.describe(session.list_sessions(proj)[0])
    assert not session.set_status(proj, "nope", "kept")


def test_sessions_all_spans_projects(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    os.makedirs(a), os.makedirs(b)
    session.save_session(a, _msgs("task in a"), {}, session_id="s1")
    session.save_session(b, _msgs("task in b"), {}, session_id="s2")
    rows = session.sessions_all()
    assert len(rows) == 2
    assert {os.path.basename(r["cwd"]) for r in rows} == {"a", "b"}
    assert rows[0]["updated"] >= rows[1]["updated"]  # newest first


def test_prune_exempts_live_worktree_sessions(tmp_path):
    proj = str(tmp_path / "proj")
    wt_dir = str(tmp_path / "livewt")
    os.makedirs(proj), os.makedirs(wt_dir)
    # Oldest session owns a LIVE worktree; then RETAIN+3 newer plain sessions.
    session.save_session(proj, _msgs("kept work"),
                         {"worktree": {"id": "w", "path": wt_dir,
                                       "branch": "chad/w", "origin_top": proj}},
                         session_id="00000000-000000-aaaa")
    # ...and one old session whose worktree is GONE — prunable like any other.
    session.save_session(proj, _msgs("dead wt"),
                         {"worktree": {"id": "x", "path": str(tmp_path / "gone"),
                                       "branch": "chad/x", "origin_top": proj}},
                         session_id="00000000-000001-bbbb")
    for i in range(session.RETAIN + 3):
        session.save_session(proj, _msgs(f"n{i}"), {},
                             session_id=f"20990101-{i:06d}-cccc")
    ids = {r["session_id"] for r in session.list_sessions(proj)}
    assert "00000000-000000-aaaa" in ids          # live worktree -> exempt
    assert "00000000-000001-bbbb" not in ids      # dead worktree -> pruned
    assert len(ids) == session.RETAIN + 1         # keep newest N + the exempt one


# -- agent resume state -------------------------------------------------------

def _agent(**kw):
    return Agent(ScriptedEngine(["never called"]), persist=False, **kw)


def test_thread_inherited_from_resume_state():
    a = _agent(resume=_msgs(), resume_state={"thread_id": "T-orig"})
    assert a.thread_id == "T-orig"
    fresh = _agent()
    assert fresh.thread_id == fresh.session_id  # a fresh session starts its thread


def test_resume_state_restores_ambient_and_todos():
    state = {"thread_id": "T", "ambient": {"calls": 5, "edited": {"a.py": []},
                                           "wrote": ["b.py"], "last_run": None},
             "todos": [{"content": "finish", "status": "in_progress"}]}
    _agent(resume=_msgs(), resume_state=state)
    assert ambient.snapshot()["calls"] == 5
    assert tools.current_todos() == state["todos"]
    _agent()  # fresh agent clears both
    assert ambient.snapshot()["calls"] == 0
    assert tools.current_todos() == []


def test_resume_system_reuses_saved_prompt():
    saved = [{"role": "system", "content": "SAVED-SYSTEM-PROMPT"},
             {"role": "user", "content": "hi"}]
    a = _agent(resume=saved, resume_system=True)
    assert a.messages[0]["content"] == "SAVED-SYSTEM-PROMPT"
    b = _agent(resume=saved)  # default: fresh system prompt, saved one dropped
    assert b.messages[0]["content"] != "SAVED-SYSTEM-PROMPT"
    assert sum(1 for m in b.messages if m["role"] == "system") == 1


def test_save_meta_carries_thread_status_ambient_todos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ambient.note_call("bash", {"command": "make"}, "[exit 0]\nok")
    tools.tool_write_todos([{"content": "x", "status": "pending"}])
    a = Agent(ScriptedEngine(["never called"]), persist=True)
    # note_call/todos above happened after Agent init (which resets) — redo them
    ambient.note_call("bash", {"command": "make"}, "[exit 0]\nok")
    tools.tool_write_todos([{"content": "x", "status": "pending"}])
    a.save()
    data = session.load_session(str(tmp_path))
    assert data["meta"]["thread_id"] == a.thread_id == a.session_id
    assert data["meta"]["status"] == "idle"
    assert data["meta"]["ambient"]["calls"] == 1
    assert data["meta"]["todos"] == [{"content": "x", "status": "pending"}]


# -- session goal (start UX) --------------------------------------------------

def test_goal_saved_and_titles_the_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = Agent(ScriptedEngine(["never called"]), persist=True)
    a.goal = "auth refactor"
    a.messages.append({"role": "user", "content": "fix the login test first"})
    a.save()
    data = session.load_session(str(tmp_path))
    assert data["meta"]["goal"] == "auth refactor"
    assert session.list_sessions(str(tmp_path))[0]["title"] == "auth refactor"


def test_goal_inherited_on_resume():
    a = _agent(resume=_msgs(), resume_state={"thread_id": "T", "goal": "ship v2"})
    assert a.goal == "ship v2"
    assert _agent().goal is None


def test_tui_goal_capture():
    from types import SimpleNamespace
    from chad.tui import TUI
    t = TUI(ScriptedEngine(["never called"]), ctx_limit=24000)
    assert t._await_goal          # fresh session asks
    t._on_accept(SimpleNamespace(text="auth refactor  "))
    assert not t._await_goal
    assert t.agent.goal == "auth refactor"
    assert not t._queue           # the answer was swallowed, not queued as a task
    t._on_accept(SimpleNamespace(text="now fix the failing test"))
    assert len(t._queue) == 1     # subsequent input is a normal task


def test_tui_goal_skip_and_commands():
    from types import SimpleNamespace
    from chad.tui import TUI
    t = TUI(ScriptedEngine(["never called"]), ctx_limit=24000)
    t._on_accept(SimpleNamespace(text=""))       # enter on empty = skip
    assert not t._await_goal and t.agent.goal is None
    t2 = TUI(ScriptedEngine(["never called"]), ctx_limit=24000)
    t2._on_accept(SimpleNamespace(text="/help"))  # a command is not a goal
    assert t2.agent.goal is None and not t2._await_goal
    t3 = TUI(ScriptedEngine(["never called"]), ctx_limit=24000,
             resume=[{"role": "user", "content": "old"}])
    assert not t3._await_goal     # resumed sessions never ask


# -- engine snapshot plumbing (pre-load path logic only) ----------------------

def test_engine_session_snapshot_paths(tmp_path):
    from chad.engine import Engine
    eng = Engine(model_id="test/model", cache_dir=str(tmp_path))
    p1 = eng._sess_ckpt_path("thread-1")
    assert os.path.basename(p1).startswith("sess-")
    assert p1 == eng._sess_ckpt_path("thread-1")           # deterministic
    assert p1 != eng._sess_ckpt_path("thread-2")           # keyed by thread
    assert p1 != Engine(model_id="test/model", kv_bits=8,
                        cache_dir=str(tmp_path))._sess_ckpt_path("thread-1")
    assert not eng.has_session_snapshot("thread-1")
    assert eng.save_session_snapshot("thread-1") == 0      # nothing resident -> skip
    assert eng.restore_session_snapshot("thread-1") == 0   # no file -> clean miss
    no_dir = Engine(model_id="test/model")
    assert not no_dir.has_session_snapshot("thread-1")     # cache_dir None -> off


def test_engine_delete_session_snapshot(tmp_path):
    from chad.engine import Engine
    eng = Engine(model_id="test/model", cache_dir=str(tmp_path))
    assert not eng.delete_session_snapshot("t")            # nothing there yet
    with open(eng._sess_ckpt_path("t"), "wb") as f:
        f.write(b"x")
    assert eng.has_session_snapshot("t")
    assert eng.delete_session_snapshot("t")                # wrap-up drops it
    assert not eng.has_session_snapshot("t")
