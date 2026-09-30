"""`cli.main` mode matrix: the permission mode, reasoning mode and resume state the
entrypoint hands to `Agent` for a given argv and terminal.

The line that matters most is the headless promotion: a one-shot task with no TTY on
stdin runs in yolo (every write and command auto-approved), because the confirm prompt
would read EOF and abort every edit. `--plan` must suppress that promotion and a TTY
must never get it — a refactor that inverts either turns no other test red.

These drive the real `_main` (real argparse, real model resolution against a local model
dir, real session store — redirected to a tmp dir by conftest) on a `cli.Host` that
reports an Apple Silicon Mac and the terminal under test, with a `cli.Backend` whose
engine, `Agent` and TUI are recorders — so no weights load and each assertion reads
exactly what `_main` decided.
"""
import json
import os
from types import SimpleNamespace

import pytest

from chad import cli, render, session


class _FakeEngine:
    """The attributes `_main` touches on an engine, and nothing else."""

    effective_ctx = 32768
    temp = 0.0

    def __init__(self, **kw):
        self.loads = 0
        self.resets = 0

    def load(self):
        self.loads += 1
        return 0.0

    def reset(self):
        self.resets += 1


@pytest.fixture
def rec(monkeypatch, tmp_path):
    """Run `cli.main` against recorders; record every engine, Agent and TUI launch.

    `rec.main(argv, tty=...)` runs the entrypoint with stdin reporting a TTY or not;
    `stdout_tty` does the same for stdout, a terminal unless a test says otherwise.
    `rec.notes` scripts the `budget_note` each successive turn banks, to drive the
    relaunch loop; left empty, every turn finishes clean. `rec.answers` are typed at the
    terminal prompts, in order. The TUI is always a recorder so a regression that falls
    through to it fails an assertion instead of taking the terminal."""
    rec = SimpleNamespace(agents=[], engines=[], notes=[], answers=[], tui=None, repl=None)

    class _RecordingAgent:
        def __init__(self, eng, **kw):
            self.eng, self.kw, self.turns = eng, kw, []
            self.budget_note = None
            self.stop_kind = None
            self.saved = False
            rec.agents.append(self)

        def run_turn(self, text):
            self.turns.append(text)
            self.budget_note = rec.notes.pop(0) if rec.notes else None
            return "ok"

        def save(self):
            self.saved = True

    def _engine(**kw):
        rec.engines.append(_FakeEngine(**kw))
        return rec.engines[-1]

    def _repl(eng, **kw):
        rec.repl = kw

    def _run_tui(eng, ctx_limit, **kw):
        rec.tui = kw

    backend = cli.Backend(engine=_engine, agent=_RecordingAgent, repl=_repl, tui=_run_tui)

    def main(argv, *, tty, stdout_tty=True):
        host = cli.Host(platform_id=lambda: ("Darwin", "arm64"),
                        stdin_isatty=lambda: tty,
                        stdout_isatty=lambda: stdout_tty,
                        ask=lambda prompt: rec.answers.pop(0))
        return cli.main(argv, host=host, load_backend=lambda: backend)

    rec.main = main

    # A local model dir, chosen the way a user would: CHAD_MODEL is the override the model
    # resolution takes, a directory has nothing to download, and its config.json is the
    # window the TUI banner reads.
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    monkeypatch.setenv("CHAD_MODEL", str(model))
    # A fixed compaction trigger, so the RAM governor never probes live Metal memory.
    monkeypatch.setenv("CHAD_CTX_LIMIT", "24000")
    # The knobs that change _main's control flow on the paths under test.
    for var in ("CHAD_AUTO_CONTINUE", "CHAD_TURN_BUDGET_S", "CHAD_REVIEW_PASS",
                "CHAD_DISABLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # the session store is keyed on cwd
    return rec


# --- the plain REPL ---------------------------------------------------------

@pytest.mark.parametrize("argv, mode", [
    (["--repl"], "normal"),
    (["--repl", "--plan"], "plan"),
    (["--repl", "--yolo"], "yolo"),
])
def test_repl_starts_in_the_mode_asked_for(rec, argv, mode):
    rec.main(argv, tty=True)
    assert rec.repl["mode"] == mode


# --- one-shot permission mode -------------------------------------------------

@pytest.mark.parametrize("argv, tty, mode, promoted", [
    (["do X"], False, "yolo", True),            # headless: promoted, and says so
    (["do X"], True, "normal", False),          # a TTY keeps the confirm prompt
    (["--plan", "do X"], False, "plan", False),  # --plan is never promoted
    (["--plan", "do X"], True, "plan", False),
    (["--yolo", "do X"], True, "yolo", False),  # explicit, so no promotion banner
])
def test_one_shot_permission_mode(rec, capsys, argv, tty, mode, promoted):
    rec.main(argv, tty=tty)

    (agent,) = rec.agents
    assert agent.kw["mode"] == mode
    assert agent.kw["yolo"] is (mode == "yolo")
    assert agent.kw["thinking"] is True
    assert agent.kw["resume"] is None
    assert agent.turns == ["do X"]
    assert agent.saved  # a follow-up `chad -c` can pick the thread up
    assert ("[headless: auto-approving" in capsys.readouterr().err) is promoted


def test_bad_argv_still_fails_in_argparse(rec):
    with pytest.raises(SystemExit) as exc:
        rec.main(["--no-such-flag", "do X"], tty=True)

    assert exc.value.code == 2
    assert rec.agents == [] and rec.engines == []


def test_no_think_turns_thinking_off(rec):
    rec.main(["--no-think", "do X"], tty=True)

    assert rec.agents[0].kw["thinking"] is False


# --- resume -------------------------------------------------------------------

def test_continue_resumes_the_newest_session(rec):
    older = [{"role": "user", "content": "older task"}]
    newest = [{"role": "user", "content": "newest task"},
              {"role": "assistant", "content": "newest answer"}]
    session.save_session(os.getcwd(), older, {}, session_id="20260101-000000-aaaa")
    session.save_session(os.getcwd(), newest, {}, session_id="20260101-000100-bbbb")

    rec.main(["-c", "do X"], tty=True)

    assert rec.agents[0].kw["resume"] == newest


def test_continue_without_a_saved_session_starts_fresh(rec, capsys):
    rec.main(["-c", "do X"], tty=True)

    assert rec.agents[0].kw["resume"] is None
    assert "no saved session" in capsys.readouterr().err


def test_resume_without_a_tty_exits_instead_of_prompting(rec, capsys):
    session.save_session(os.getcwd(), [{"role": "user", "content": "t"}], {})

    with pytest.raises(SystemExit) as exc:
        rec.main(["--resume"], tty=False)

    assert exc.value.code == 1
    assert rec.agents == [] and rec.tui is None
    assert "needs an interactive terminal" in capsys.readouterr().err


def test_resume_on_a_tty_loads_the_picked_session_not_the_newest(rec):
    older = [{"role": "user", "content": "older task"}]
    session.save_session(os.getcwd(), older, {}, session_id="20260101-000000-aaaa")
    session.save_session(os.getcwd(), [{"role": "user", "content": "newer"}], {},
                         session_id="20260101-000100-bbbb")
    rec.answers = ["2"]  # the pick list is newest first, so #2 is the older session

    rec.main(["--resume", "do X"], tty=True)

    assert rec.answers == []  # the prompt was actually asked
    assert rec.agents[0].kw["resume"] == older


# --- no task: the TUI ---------------------------------------------------------

@pytest.mark.parametrize("argv, mode", [
    ([], "normal"),
    (["--plan"], "plan"),
    (["--yolo"], "yolo"),
])
def test_no_task_launches_the_tui_in_the_requested_mode(rec, argv, mode):
    rec.main(argv, tty=True)

    assert rec.agents == []  # the TUI builds its own Agent
    assert rec.tui["mode"] == mode
    assert rec.tui["thinking"] is True and rec.tui["resume"] is None
    assert rec.engines[0].loads == 0  # weights load on the TUI's thread, via finalize


# --- one-shot relaunch after a budget stop ------------------------------------

def test_headless_budget_stop_relaunches_fresh_with_the_progress_note(rec, monkeypatch):
    monkeypatch.setenv("CHAD_TEMP", "0")  # greedy, so the relaunch has to raise it
    rec.notes = ["Progress so far: edited a.py", None]

    rec.main(["do X"], tty=False)

    first, second = rec.agents
    assert second.kw["mode"] == "yolo" and second.kw["yolo"] is True
    assert "resume" not in second.kw  # a fresh context, not a replay of the stuck one
    assert second.turns == ["do X\n\n[Progress so far: edited a.py]"]
    assert rec.engines[0].resets == 1
    assert rec.engines[0].temp == 0.6  # a greedy stall would replay itself verbatim
    assert second.saved and not first.saved


def test_headless_relaunches_stop_after_two(rec):
    rec.notes = ["note 1", "note 2", "note 3", "note 4"]

    rec.main(["do X"], tty=False)

    assert len(rec.agents) == 3
    assert rec.agents[-1].budget_note == "note 3"  # still banked; the run gives up
    assert rec.agents[-1].saved


def test_interactive_budget_stop_does_not_relaunch(rec):
    rec.notes = ["Progress so far: nothing landed"]

    rec.main(["do X"], tty=True)

    assert len(rec.agents) == 1 and rec.agents[0].saved
    assert rec.engines[0].resets == 0


# --- the exit status a script reads -------------------------------------------

def test_headless_clean_finish_exits_zero(rec):
    assert rec.main(["do X"], tty=False) == 0


def test_headless_exhausted_relaunches_exit_not_done(rec, monkeypatch):
    monkeypatch.setenv("CHAD_AUTO_CONTINUE", "1")
    rec.notes = ["n1", "n2"]

    assert rec.main(["do X"], tty=False) == 1
    assert len(rec.agents) == 2


def test_headless_recovered_relaunch_exits_zero(rec, monkeypatch):
    monkeypatch.setenv("CHAD_AUTO_CONTINUE", "1")
    rec.notes = ["n1"]

    assert rec.main(["do X"], tty=False) == 0
    assert len(rec.agents) == 2


@pytest.mark.parametrize("result, stop_kind, budget_note, status", [
    ("Done.", None, None, 0),
    ("[interrupted]", None, None, 130),
    ("[stopped: the same call three times]", "loop", None, 1),
    ("[budget] note", "budget", "note", 1),
])
def test_exit_status_reads_how_the_turn_ended(result, stop_kind, budget_note, status):
    assert cli._exit_status(result, stop_kind, budget_note) == status


# --- stdout piped or redirected -----------------------------------------------

def test_piped_stdout_carries_only_the_answer(rec, capsys):
    rec.main(["do X"], tty=False, stdout_tty=False)

    assert rec.agents[0].kw["emit"] is render.piped_emit
    assert capsys.readouterr().out == "ok\n"


def test_terminal_stdout_keeps_the_streaming_emitter(rec, capsys):
    rec.main(["do X"], tty=False, stdout_tty=True)

    assert rec.agents[0].kw["emit"] is None
    assert "ok" not in capsys.readouterr().out
