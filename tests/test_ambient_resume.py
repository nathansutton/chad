"""The ambient facts survive a resume.

`ambient` is module state for one session, reset whenever an Agent is built. A resumed
conversation restored its messages but not these facts, so it recalled no pre-edit test
baseline and decorated files the transcript already explained — behaviour the same
session, uninterrupted, never had. `Agent.save` now records `ambient.snapshot()` in the
session file's `meta["ambient"]`, and a resumed Agent restores it.
"""
import json
import os

import pytest

from chad import ambient, session
from chad.agent import Agent
from test_agent_e2e import ScriptedEngine
from test_ambient import _seq


@pytest.fixture
def on(monkeypatch):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    ambient.reset()


@pytest.fixture
def srcfile(tmp_path, monkeypatch):
    p = tmp_path / "mod.py"
    p.write_text("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n\n\n"
                 "class Gamma:\n    def m(self):\n        return 3\n")
    monkeypatch.chdir(tmp_path)  # repomap.service() re-roots on cwd change
    return "mod.py"


def test_snapshot_restore_keeps_the_baseline_and_the_shown_skeletons(on, srcfile):
    # The session took its pre-edit baseline, edited, and was shown one file's skeleton;
    # then it ended. The resumed session (a fresh reset) recalls the baseline on the
    # failing run and does not decorate the same file again.
    _seq(("bash", {"command": "pytest -q"}, "[exit 1]\n  2 failed\n  10 passed"),
         ("edit", {"path": "a.py"}, "[edited a.py]"))
    first = ambient.annotate("bash", {"command": f"cat {srcfile}"}, "def alpha(): ...")
    assert f"[file] {srcfile}:" in first
    saved = json.loads(json.dumps(ambient.snapshot()))   # exactly what a session file holds

    ambient.reset()
    assert ambient._baselines == {} and ambient._skeleton_shown == set()
    ambient.restore(saved)
    assert "pytest" in ambient._baselines
    assert ambient._edited == {"a.py": set()}
    again = ambient.annotate("bash", {"command": f"cat {srcfile}"}, "def alpha(): ...")
    assert "[file]" not in again                          # already shown: no second skeleton
    out = ambient.annotate("bash", {"command": "pytest -q"}, "[exit 1]\n  3 failed")
    assert "[baseline] before your first edit, `pytest` → exit 1" in out
    assert '"2 failed · 10 passed"' in out


def test_restore_skips_what_it_does_not_recognise(on):
    ambient.restore({"calls": "many", "edited": ["a.py"], "wrote": "a.py",
                     "last_run": {"head": "pytest"}, "baselines": {"pytest": "exit 1"},
                     "skeleton_shown": "a.py", "unknown": 1})
    assert ambient._calls == 0 and ambient._edited == {} and ambient._wrote == []
    assert ambient._last_run is None and ambient._baselines == {}
    assert ambient._skeleton_shown == set()
    ambient.restore({})
    assert ambient.snapshot() == {"calls": 0, "edited": {}, "wrote": [], "last_run": None,
                                  "baselines": {}, "skeleton_shown": []}


def test_the_definition_pointer_memo_is_not_carried_over(on):
    ambient._def_pointer_seen["alpha"] = ""        # "nowhere", as of the ended session
    assert "def_pointer_seen" not in ambient.snapshot()


def _agent(persist=False, resume=None, resume_meta=None):
    eng = ScriptedEngine([])
    return Agent(eng, mode="yolo", thinking=False, persist=persist, resume=resume,
                 resume_meta=resume_meta)


def test_save_records_the_facts_and_a_resume_restores_them(tmp_path, monkeypatch, on):
    monkeypatch.chdir(tmp_path)
    agent = _agent(persist=True)
    _seq(("bash", {"command": "pytest -q"}, "[exit 1]\n  1 failed"),
         ("write", {"path": "b.py"}, "[wrote b.py]"))
    agent.messages.append({"role": "user", "content": "hello"})
    agent.save()

    data = session.load_session(os.getcwd())
    assert data["meta"]["ambient"]["wrote"] == ["b.py"]
    assert "pytest" in data["meta"]["ambient"]["baselines"]

    # A fresh Agent resets the facts; one seeded from the saved session has them back.
    _agent()
    assert ambient._wrote == [] and ambient._baselines == {}
    _agent(resume=data["messages"], resume_meta=data["meta"])
    assert ambient._wrote == ["b.py"] and "pytest" in ambient._baselines
    assert ambient._calls == 2


def test_a_resume_without_saved_facts_starts_clean(on):
    _seq(("write", {"path": "b.py"}, "[wrote b.py]"))
    _agent(resume=[{"role": "user", "content": "x"}], resume_meta={"mode": "normal"})
    assert ambient._wrote == []
    _seq(("write", {"path": "b.py"}, "[wrote b.py]"))
    _agent(resume=[{"role": "user", "content": "x"}], resume_meta=None)
    assert ambient._wrote == []
