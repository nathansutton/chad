"""Every registered lever must actually change behavior when switched off.

This is the test that makes `evals/ablate.py` trustworthy. A lever whose guard is
misplaced — gating a branch that never runs, or reverting to a state identical to the
fix — produces a per-lever delta of zero. Read off the ablation table, that says "this
fix does nothing, drop it", and the fix gets deleted while the bug it prevented comes
back. A decorative gate is worse than no gate, because it launders a measurement error
into a confident conclusion.

So: for each lever, one minimal pair. Same input, `CHAD_DISABLE` unset vs set, different
observable output. The final test asserts the coverage set equals the registry, so a new
lever cannot be added without proving it bites.
"""
import os

import pytest

from chad import compaction, guardrails, levers, profiles, syntaxgate, tools

# Every lever exercised below. The last test asserts this equals levers.LEVERS.
COVERED = set()


def bite(name):
    """Mark a lever as having a minimal on/off pair in this file."""
    COVERED.add(name)
    return name


def off(monkeypatch, *names):
    monkeypatch.setenv("CHAD_DISABLE", ",".join(names))


def on(monkeypatch):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)


# === iter-2 ================================================================

def test_verify_requires_execution(monkeypatch):
    """A display-only command exiting 0 is not verification — unless ablated."""
    n = bite("verify_requires_execution")
    on(monkeypatch)
    assert not guardrails.bash_result_verifies("some output", "sed -n '1,5p' f.py | cat -A")
    off(monkeypatch, n)
    assert guardrails.bash_result_verifies("some output", "sed -n '1,5p' f.py | cat -A")


def _nudge(text, **kw):
    args = dict(hit_cap=False, made_edit=False, unverified_edit=False,
                read_only_intent=False, action_task=True, truncation_nudges=0,
                answer_nudges=0, verify_nudges=0, open_tool_call=False)
    args.update(kw)
    return guardrails.nudge_for_no_calls(text, **args)


def test_bail_nudge(monkeypatch):
    """Empty content after </think> must be nudged, not accepted as a final answer."""
    n = bite("bail_nudge")
    text = "<think>I should look at the file</think>"
    on(monkeypatch)
    kind, msg = _nudge(text)
    assert kind == "no-edit" and "no tool call" in msg
    off(monkeypatch, n)
    kind2, msg2 = _nudge(text)
    # Falls through to the answered-on-paper branch (action_task), whose text differs.
    assert msg2 != msg


def test_done_spec_recheck(monkeypatch):
    """Fires once, only on a real action turn (work done, verified, not explain-only)."""
    n = bite("done_spec_recheck")
    on(monkeypatch)
    assert levers.enabled(n)
    # Happy path: fires.
    assert guardrails.done_spec_recheck(
        did_work=True, unverified_edit=False, recheck_done=False, read_only_intent=False)
    # Already fired this turn -> don't repeat.
    assert not guardrails.done_spec_recheck(
        did_work=True, unverified_edit=False, recheck_done=True, read_only_intent=False)
    # No work / unverified / explain-only -> other gates own those; recheck stays silent.
    assert not guardrails.done_spec_recheck(
        did_work=False, unverified_edit=False, recheck_done=False, read_only_intent=False)
    assert not guardrails.done_spec_recheck(
        did_work=True, unverified_edit=True, recheck_done=False, read_only_intent=False)
    assert not guardrails.done_spec_recheck(
        did_work=True, unverified_edit=False, recheck_done=False, read_only_intent=True)
    off(monkeypatch, n)
    assert not levers.enabled(n)


def test_recheck_spiral():
    """Plan 070: the post-recheck edit cap trips only once landed fix edits exceed
    RECHECK_MAX_FIX_EDITS — one or two targeted fixes are fine, a run of them is a
    thrash on already-correct work."""
    assert not guardrails.recheck_spiral(0)
    assert not guardrails.recheck_spiral(guardrails.RECHECK_MAX_FIX_EDITS)
    assert guardrails.recheck_spiral(guardrails.RECHECK_MAX_FIX_EDITS + 1)


def test_open_tool_call_nudged_without_cap():
    """Iter-3: an unbalanced tool-call attempt that parsed to zero calls is never a final
    answer — even when the token cap was NOT hit (a sampling glitch / premature EOS, e.g.
    TB2 vulnerable-secret's 28-token `{"name":"bash>` garble). It must be nudged, not
    accepted, and the message must name the malformed-call cause (not the length limit)."""
    kind, msg = _nudge('<tool_call>{"name": "bash>', hit_cap=False, open_tool_call=True)
    assert kind == "truncated" and "malformed" in msg
    # hit_cap + open call still gets the write-in-parts guidance (length limit).
    kind2, msg2 = _nudge("<tool_call>{...", hit_cap=True, open_tool_call=True)
    assert kind2 == "truncated" and "length limit" in msg2
    # bounded: once truncation_nudges hits 2, stop nudging.
    kind3, _ = _nudge('<tool_call>{"name": "bash>', hit_cap=False,
                      open_tool_call=True, truncation_nudges=2)
    assert kind3 != "truncated"


def test_investigation_gate(monkeypatch):
    n = bite("investigation_gate")
    on(monkeypatch)
    assert guardrails.investigation_gate(8, made_edit=False, gate_nudges=0)
    off(monkeypatch, n)
    assert guardrails.investigation_gate(8, made_edit=False, gate_nudges=0) is None


def test_edit_loop_break(monkeypatch):
    n = bite("edit_loop_break")
    on(monkeypatch)
    assert guardrails.edit_loop_break(2, 0, kind="nomatch")
    off(monkeypatch, n)
    assert guardrails.edit_loop_break(2, 0, kind="nomatch") is None


def test_grep_zero_match_notice(monkeypatch, tmp_path):
    n = bite("grep_zero_match_notice")
    (tmp_path / "a.py").write_text("nothing here\n")
    on(monkeypatch)
    hot = tools.tool_grep("zzz_no_such_symbol", str(tmp_path))
    assert "no matches for" in hot and str(tmp_path) in hot, "must state its scope"
    off(monkeypatch, n)
    cold = tools.tool_grep("zzz_no_such_symbol", str(tmp_path))
    assert cold == "[no matches]", "ablated: the bare, scope-free claim"


def test_syntaxgate_revert(monkeypatch, tmp_path):
    n = bite("syntaxgate_revert")
    p = str(tmp_path / "m.py")
    before, after = "def f():\n    return 1\n", "def f():\n        return 1\n  x=2\n"
    on(monkeypatch)
    assert syntaxgate.indent_reject(p, before, after), "indent break must be rejected"
    off(monkeypatch, n)
    assert syntaxgate.indent_reject(p, before, after) is None, "ablated: the break lands"


def test_structural_reindent(monkeypatch, tmp_path):
    n = bite("structural_reindent")
    p = str(tmp_path / "e.py")
    before = "def f(self):\n    x = 1\n    return x\n"
    # A two-LEVEL block the model mis-indents (comment fine, if over-indented, body under):
    # only the structural reindent can fix it — fit preserves the garbage, snap flattens
    # the if-body into the if (still broken).
    new = "x = 1\n# note\n      if x:\n  y = 2"
    on(monkeypatch)
    open(p, "w").write(before)
    assert "reindented to structure" in tools.tool_replace_lines(p, 2, 2, new)
    off(monkeypatch, n)
    open(p, "w").write(before)
    assert tools.tool_replace_lines(p, 2, 2, new).startswith("[edit rejected"), \
        "ablated: no structural reindent, and fit/snap can't fix a multi-level block"


# === iter-3 ================================================================

def test_progress_note_rich(monkeypatch):
    n = bite("progress_note_rich")
    msgs = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content":
            "The bug is that apply_tiered_discount uses > instead of >=.\n"
            '<tool_call>{"name": "read", "arguments": {"path": "pricing.py"}}</tool_call>'},
        {"role": "tool", "name": "bash", "content": "[exit 1]\nAssertionError: 0.9 != 0.8"},
    ]
    on(monkeypatch)
    rich = guardrails.progress_note(msgs)
    assert "Working hypothesis" in rich
    assert "AssertionError" in rich, "the failing-check signature must survive"
    assert "Already examined" in rich
    off(monkeypatch, n)
    thin = guardrails.progress_note(msgs)
    for gone in ("Working hypothesis", "AssertionError", "Already examined"):
        assert gone not in thin, f"ablated note must drop {gone!r}"


def test_grep_filter_before_cap(monkeypatch, tmp_path):
    """A dir-heavy tree must not exhaust the file budget on entries that can't match."""
    n = bite("grep_filter_before_cap")
    for i in range(12):                        # directories the walk also yields
        (tmp_path / f"locale_{i:02d}").mkdir()
    (tmp_path / "zz_target.py").write_text("def needle():\n    pass\n")
    monkeypatch.setattr(tools, "GREP_MAX_FILES", 6)
    on(monkeypatch)
    hot = tools.tool_grep("needle", str(tmp_path))
    assert "zz_target.py" in hot, "filtering first must reach the real file"
    off(monkeypatch, n)
    cold = tools.tool_grep("needle", str(tmp_path))
    # Not `"needle" not in cold` — the zero-match notice quotes the pattern back.
    assert "zz_target.py" not in cold, "ablated: the budget is eaten by directories"
    assert "only the first 6 files were searched" in cold, \
        "and at least it still admits the truncation (that's a different lever)"


def test_repeat_coarse_tier(monkeypatch):
    n = bite("repeat_coarse_tier")
    block = ("I need to examine the configuration more carefully. " * 20) + "\n"
    text = block * 12                       # ~13KB of a repeated paragraph-scale unit
    on(monkeypatch)
    assert guardrails.degenerate_tail(text), "block-scale loop must be caught"
    off(monkeypatch, n)
    assert not guardrails.degenerate_tail(text), "ablated: fine tier is blind to it"


def test_edit_fail_kind(monkeypatch):
    """A no-op edit must not be told to go re-read and paste verbatim."""
    n = bite("edit_fail_kind")
    on(monkeypatch)
    hot = guardrails.edit_loop_break(2, 0, kind="noop")
    assert "identical" in hot and "will not help" in hot
    off(monkeypatch, n)
    cold = guardrails.edit_loop_break(2, 0, kind="noop")
    assert "verbatim" in cold, "ablated: the pre-iter-3 conflation returns"
    assert cold != hot


def test_revert_rearm_gate(monkeypatch):
    n = bite("revert_rearm_gate")
    args = {"command": "git checkout -- src/mod.py"}
    on(monkeypatch)
    _, made, unver = guardrails.update_work_flags(
        "bash", args, "ok", did_work=True, made_edit=True, unverified_edit=True)
    assert not made and not unver, "a clean revert un-lands the edit"
    off(monkeypatch, n)
    _, made2, _ = guardrails.update_work_flags(
        "bash", args, "ok", did_work=True, made_edit=True, unverified_edit=True)
    assert made2, "ablated: revert-then-prose can ship an empty diff"


def test_subagent_compact_window(monkeypatch):
    """Pass 3 must not be dead code in a sub-agent (one user message, at index 1)."""
    n = bite("subagent_compact_window")

    def transcript():
        m = [{"role": "system", "content": "S" * 2000},
             {"role": "user", "content": "find the retry handling"}]
        for i in range(6):
            m.append({"role": "assistant", "content": f"<think>r{i}</think>grep"})
            m.append({"role": "tool", "name": "grep", "content": "hit " * 900})
        return m

    def render_for(m):
        return lambda: list("".join(x["content"] for x in m))

    def n_deleted(disable):
        if disable:
            os.environ["CHAD_DISABLE"] = disable
        else:
            os.environ.pop("CHAD_DISABLE", None)
        msgs = transcript()
        r = render_for(msgs)
        before = len(msgs)
        compaction.compact_if_needed(msgs, r, lambda *a: None,
                                     ctx_limit=len(r()) // 2, prompt_ids=r())
        # the notice is appended, so count real deletions rather than net length
        return before - sum(1 for m in msgs if compaction._NOTICE_TAG not in m["content"])

    monkeypatch.setenv("CHAD_OFFLOAD_DIR", "/tmp/chad-bite-offload")
    hot = n_deleted(None)
    cold = n_deleted(f"{n},compact_notice,compact_offload")
    assert hot > 0, "pass 3 must shed old churn in a sub-agent"
    assert cold == 0, "ablated: the deletable range collapses and pass 3 is dead code"


def test_backend_retry(monkeypatch, tmp_path):
    n = bite("backend_retry")
    from chad.agent import Agent
    from chad.base_engine import BackendError
    from test_agent_e2e import ScriptedEngine, _tool_call

    class FlakyEngine(ScriptedEngine):
        """Fails the first generate with a transient 5xx, then behaves."""

        def __init__(self, script):
            super().__init__(script)
            self.raised = False

        def generate(self, *a, **kw):
            if not self.raised:
                self.raised = True
                raise BackendError("llama-server HTTP 500: boom", transient=True)
            return super().generate(*a, **kw)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.py").write_text("x = 1\n")
    monkeypatch.setattr("chad.agent.time.sleep", lambda *_: None)
    # `done` alone is rejected for having done no work, which would eat a scripted turn
    # and mask the retry — give the loop one real tool call first.
    script = [_tool_call("read", path="m.py"), _tool_call("done", summary="ok")]

    on(monkeypatch)
    ag = Agent(FlakyEngine(script), mode="auto", thinking=False)
    assert ag.run_turn("do it", stream=False), "a transient 5xx must be re-rolled"

    off(monkeypatch, n)
    ag = Agent(FlakyEngine(script), mode="auto", thinking=False)
    with pytest.raises(BackendError):
        ag.run_turn("do it", stream=False)


def test_subagent_no_respawn(monkeypatch, tmp_path):
    n = bite("subagent_no_respawn")
    from chad.agent import Agent
    from test_agent_e2e import ScriptedEngine, _tool_call
    monkeypatch.chdir(tmp_path)

    def run(disable):
        if disable:
            monkeypatch.setenv("CHAD_DISABLE", disable)
        else:
            monkeypatch.delenv("CHAD_DISABLE", raising=False)
        spawns = []
        ag = Agent(ScriptedEngine([
            _tool_call("task", description="find it", prompt="where is retry handled?"),
            _tool_call("task", description="find it", prompt="where is retry handled?"),
            _tool_call("done", summary="ok"),
        ]), mode="auto", thinking=False)
        ag._run_subagent = lambda d, p, *a, **k: (spawns.append((d, p)), "found nothing")[1]
        ag.run_turn("add retry handling", stream=False)
        return spawns, ag

    spawns, ag = run(None)
    assert len(spawns) == 1, "the identical re-spawn must be refused"
    assert any("already ran this exact sub-agent" in m.get("content", "")
               for m in ag.messages)
    spawns, _ = run(n)
    assert len(spawns) == 2, "ablated: the duplicate runs and burns the budget"


def test_subagent_budget_note(monkeypatch, tmp_path):
    """A sub-agent that dies must still hand back where it got to. Driven for real: the
    sub's turn raises, the parent catches it, and the salvage path is the only thing
    standing between the parent and a bare '[task failed: …]' sentinel."""
    n = bite("subagent_budget_note")
    from chad.agent import Agent
    from test_agent_e2e import ScriptedEngine, _tool_call
    monkeypatch.chdir(tmp_path)

    class DyingSubEngine(ScriptedEngine):
        """Generate #1 = the parent spawning the task; #2 = the sub-agent, which dies."""

        def generate(self, *a, **kw):
            self._n = getattr(self, "_n", 0) + 1
            if self._n == 2:
                raise RuntimeError("sub-agent exploded")
            return super().generate(*a, **kw)

    def run(disable):
        if disable:
            monkeypatch.setenv("CHAD_DISABLE", disable)
        else:
            monkeypatch.delenv("CHAD_DISABLE", raising=False)
        ag = Agent(DyingSubEngine([
            _tool_call("task", description="find it", prompt="where is retry handled?"),
            "unused",                       # consumed by the sub-agent's raising turn
            _tool_call("done", summary="ok"),
        ]), mode="auto", thinking=False)
        ag.run_turn("add retry handling", stream=False)
        return "\n".join(m.get("content", "") for m in ag.messages if m.get("name") == "task")

    hot = run(None)
    assert "task failed" in hot and "sub-agent progress before it stopped" in hot, \
        "a dead sub-agent's findings must be salvaged, not discarded"
    cold = run(n)
    assert "task failed" in cold
    assert "sub-agent progress" not in cold, "ablated: the findings die with the sub-agent"


# === playbook levers (behavior asserted in their own suites) ===============

def test_playbook_levers_have_dedicated_suites(monkeypatch):
    """compact_notice / compact_offload live in test_compact_notice.py, plan_review in
    test_plan_review.py, profile_prompt in test_levers.py. Assert the guards exist and
    respond, so the coverage check below is honest rather than a rubber stamp."""
    for n in ("compact_notice", "compact_offload", "plan_review"):
        bite(n)
        on(monkeypatch)
        assert levers.enabled(n)
        off(monkeypatch, n)
        assert not levers.enabled(n)
    bite("profile_prompt")
    monkeypatch.setenv("CHAD_PROFILE", "ornith")
    on(monkeypatch)
    assert profiles.prompt_block(None) != ""
    off(monkeypatch, "profile_prompt")
    assert profiles.prompt_block(None) == ""


# === the coverage contract =================================================

def test_every_registered_lever_has_a_bite_test():
    """A lever with no minimal on/off pair is a lever that might not bite. Ablating it
    would report 'no measured effect' and the fix would be deleted on that evidence."""
    missing = set(levers.LEVERS) - COVERED
    assert not missing, f"levers with no bite test: {sorted(missing)}"
    stale = COVERED - set(levers.LEVERS)
    assert not stale, f"bite tests for unregistered levers: {sorted(stale)}"
