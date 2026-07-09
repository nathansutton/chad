"""Tests for the plan-then-review injection (lever: plan_review).

Plan mode's preamble already demands a Context section, exact file paths with
current-state excerpts, numbered steps, verify commands, conventions, and out-of-scope —
and the model still ships plans missing half of it. A standing rule in the preamble has
washed out by the time the plan is actually being written.

So the ask is delivered as a SECOND message, after the artifact exists: write the plan,
then get told to go re-read it against the rubric. Same placement lesson as the
keep-reading notice, which did nothing in `read`'s tool description and worked the moment
it moved into `read`'s return value.

Driven through the real `run_turn` loop with the scripted engine from test_agent_e2e —
no weights, no GPU, no network.
"""
import os

from chad.agent import Agent
from test_agent_e2e import ScriptedEngine, _tool_call

_PLAN_BODY = "# Context\nwhy\n# Steps\n1. do it\n# Verify\npytest -q\n"


def _plan_agent(script, **kw):
    return Agent(ScriptedEngine(script), mode="plan", thinking=False, **kw)


def _reviews(agent):
    return [m for m in agent.messages
            if "Before you call `done`" in m.get("content", "")]


def test_review_is_injected_after_the_plan_is_written(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.chdir(tmp_path)
    agent = _plan_agent([
        _tool_call("write", path="plans/001-retry.md", content=_PLAN_BODY),
        _tool_call("done", summary="plan written"),
    ])
    agent.run_turn("add retry handling to the fetch client", stream=False)

    assert len(_reviews(agent)) == 1, "the review ask must be delivered exactly once"
    # Placement is the mechanism: the ask must land AFTER the write result, not before.
    idx_write = next(i for i, m in enumerate(agent.messages)
                     if m.get("name") == "write" and "[wrote" in m.get("content", ""))
    idx_review = next(i for i, m in enumerate(agent.messages)
                      if "Before you call `done`" in m.get("content", ""))
    assert idx_review == idx_write + 1, "review must sit next to the artifact it reviews"

    body = _reviews(agent)[0]["content"]
    assert os.path.abspath("plans/001-retry.md") in body, "name the file it must re-read"
    for item in ("Context", "current-state", "numbered", "verify", "out-of-scope"):
        assert item in body, f"rubric is missing {item!r}"
    assert "WITHOUT access to this conversation" in body


def test_review_fires_once_not_per_edit(monkeypatch, tmp_path):
    """The model is told to fix the file with `edit`. That edit must not re-trigger the
    review and loop the turn."""
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.chdir(tmp_path)
    agent = _plan_agent([
        _tool_call("write", path="plans/001-a.md", content=_PLAN_BODY),
        _tool_call("edit", path="plans/001-a.md", old="do it", new="do it precisely"),
        _tool_call("write", path="plans/002-b.md", content=_PLAN_BODY),
        _tool_call("done", summary="done"),
    ])
    agent.run_turn("add retry handling to the fetch client", stream=False)
    assert len(_reviews(agent)) == 1


def test_no_review_when_no_plan_was_written(monkeypatch, tmp_path):
    """A conceptual question in plan mode answers in prose and writes no plan file. There
    is no artifact to review, so there must be no ask."""
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")
    agent = _plan_agent([
        _tool_call("read", path="mod.py"),
        _tool_call("done", summary="it returns 1"),
    ])
    agent.run_turn("what does mod.py do?", stream=False)
    assert not _reviews(agent)


def test_review_is_gated_by_its_lever(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAD_DISABLE", "plan_review")
    monkeypatch.chdir(tmp_path)
    agent = _plan_agent([
        _tool_call("write", path="plans/001-retry.md", content=_PLAN_BODY),
        _tool_call("done", summary="plan written"),
    ])
    agent.run_turn("add retry handling to the fetch client", stream=False)
    assert not _reviews(agent)
    assert os.path.exists("plans/001-retry.md"), "the plan itself is unaffected"
