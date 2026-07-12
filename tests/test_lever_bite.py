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
    p2 = str(tmp_path / "e2.py")   # fresh path: keep the stale-file guard out of this arm
    open(p2, "w").write(before)
    assert tools.tool_replace_lines(p2, 2, 2, new).startswith("[edit rejected"), \
        "ablated: no structural reindent, and fit/snap can't fix a multi-level block"


# === iter-6 (plan 073) =====================================================

def test_syntax_revert(monkeypatch, tmp_path):
    """The 073 corruption engine: replacing ONE physical line of a multi-line def
    signature with a fragment. ON: rejected, file intact. OFF: warn-and-land."""
    n = bite("syntax_revert")
    p = str(tmp_path / "s.py")
    before = "def generate(\n        prompt_ids,\n        max_tokens,\n):\n    return 1\n"
    on(monkeypatch)
    open(p, "w").write(before)
    res = tools.tool_replace_lines(p, 2, 2, "def generate(")
    assert res.startswith("[edit rejected") and "unparseable" in res
    assert open(p).read() == before, "reject must leave the file byte-identical"
    off(monkeypatch, n)
    p2 = str(tmp_path / "s2.py")   # fresh path: keep the stale-file guard out of this arm
    open(p2, "w").write(before)
    res = tools.tool_replace_lines(p2, 2, 2, "def generate(")
    assert res.startswith("[edited") and "no longer parses" in res, \
        "ablated: the severing edit lands with only a warning"


def test_edit_result_echo(monkeypatch, tmp_path):
    n = bite("edit_result_echo")
    p = str(tmp_path / "r.py")
    before = "a = 1\nb = 2\nc = 3\n"
    on(monkeypatch)
    open(p, "w").write(before)
    res = tools.tool_replace_lines(p, 2, 2, "b = 20\nbb = 21")
    assert "use THESE numbers" in res and "shifted by +1" in res
    off(monkeypatch, n)
    p2 = str(tmp_path / "r2.py")   # fresh path: keep the stale-file guard out of this arm
    open(p2, "w").write(before)
    res = tools.tool_replace_lines(p2, 2, 2, "b = 20\nbb = 21")
    assert res.startswith("[edited") and "use THESE numbers" not in res


def test_stale_file_guard(monkeypatch, tmp_path):
    """Line numbers minted before an out-of-band change are blind: reject once with a
    fresh view, then allow the re-send (the reject itself refreshes the anchor)."""
    n = bite("stale_file_guard")
    p = str(tmp_path / "g.py")
    on(monkeypatch)
    open(p, "w").write("a = 1\nb = 2\n")
    tools.tool_read(p)                                  # the model saw this content
    open(p, "w").write("# moved\na = 1\nb = 2\n")       # out-of-band change (sed/git)
    res = tools.tool_replace_lines(p, 2, 2, "b = 20")
    assert res.startswith("[edit rejected") and "changed on disk" in res
    res = tools.tool_replace_lines(p, 3, 3, "b = 20")   # re-send with fresh numbers
    assert res.startswith("[edited")
    off(monkeypatch, n)
    open(p, "w").write("a = 1\nb = 2\n")
    tools.tool_read(p)
    open(p, "w").write("# moved\na = 1\nb = 2\n")
    assert tools.tool_replace_lines(p, 3, 3, "b = 20").startswith("[edited"), \
        "ablated: the stale edit goes straight through"


# === iter-7 (plan 074) =====================================================

def test_edit_drift_warn(monkeypatch):
    """The measured drift: a rewrite drops a def still called elsewhere. ON: the
    result warns with the dangling reference. OFF: silent (the --context-tokens
    AttributeError class ships without a word)."""
    n = bite("edit_drift_warn")
    before = "def helper():\n    return 1\n\ndef main():\n    return helper()\n"
    after = "def main():\n    return helper()\n"
    on(monkeypatch)
    assert syntaxgate.drift_warn("m.py", before, after)
    off(monkeypatch, n)
    assert syntaxgate.drift_warn("m.py", before, after) is None


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


# === iter-8 (plan 079) =====================================================

def test_ts_edit_revert(monkeypatch, tmp_path):
    """The vm.js/ars.R class: a targeted edit breaks a clean non-Python file. ON:
    rejected, file intact. OFF: warn-and-land — the corruption that compounded through
    6-20 follow-up edits to reward-zero benchmark tasks."""
    n = bite("ts_edit_revert")
    before = "int main(){ return 0; }\n"
    on(monkeypatch)
    p = tmp_path / "a.c"
    p.write_text(before)
    res = tools.tool_edit(str(p), "return 0;", "return 0")   # drop the semicolon
    assert res.startswith("[edit rejected") and "unparseable" in res
    assert p.read_text() == before, "reject must leave the file byte-identical"
    off(monkeypatch, n)
    p2 = tmp_path / "b.c"
    p2.write_text(before)
    res = tools.tool_edit(str(p2), "return 0;", "return 0")
    assert res.startswith("[edited") and "warning" in res, \
        "ablated: the non-Python break lands with only a warning"


def test_write_gate(monkeypatch, tmp_path):
    """Whole-file write refusing content that newly breaks the parse. ON: rejected,
    disk untouched — and an already-broken file stays overwritable (the repair path).
    OFF: the warn-only write that delivered 51/55 benchmark landed breaks."""
    n = bite("write_gate")
    on(monkeypatch)
    p = tmp_path / "w.py"
    p.write_text("x = 1\n")
    res = tools.tool_write(str(p), "def f(:\n")
    assert res.startswith("[write rejected") and "YOUR content" in res
    assert p.read_text() == "x = 1\n", "reject must leave the file untouched"
    broken = tmp_path / "broken.py"
    broken.write_text("def g(:\n")
    assert tools.tool_write(str(broken), "def g():\n    return 1\n").startswith("[wrote"), \
        "an already-broken file must stay overwritable — that IS the repair path"
    off(monkeypatch, n)
    p2 = tmp_path / "w2.py"
    p2.write_text("x = 1\n")
    res = tools.tool_write(str(p2), "def f(:\n")
    assert res.startswith("[wrote") and "no longer parses" in res, \
        "ablated: warn-and-land returns on the write path"


def test_broken_streak_steer(monkeypatch, tmp_path):
    """Two consecutive landings on a still-broken file escalate the warning to a
    whole-file-rewrite steer (the measured 14-landing streak rode flat per-edit
    warnings). OFF: every landing gets the same flat warning."""
    n = bite("broken_streak_steer")

    def land_two(p):
        p.write_text("def f(:\n")   # broken out-of-band, so edits stay allowed
        r1 = tools.tool_edit(str(p), "def f(:", "def f(:  # try1")
        r2 = tools.tool_edit(str(p), "# try1", "# try2")
        return r1, r2

    on(monkeypatch)
    r1, r2 = land_two(tmp_path / "s.py")
    assert "no longer parses" in r1 and "consecutive" not in r1, "first landing: flat warn"
    assert "consecutive" in r2 and "STOP patching" in r2, "second landing: escalate"
    off(monkeypatch, n)
    r1, r2 = land_two(tmp_path / "s2.py")
    assert "no longer parses" in r2 and "consecutive" not in r2, \
        "ablated: the streak never escalates"


def test_write_diff_note(monkeypatch, tmp_path):
    n = bite("write_diff_note")
    p = tmp_path / "d.py"
    on(monkeypatch)
    p.write_text("a = 1\nb = 2\n")
    res = tools.tool_write(str(p), "a = 1\nb = 3\n")
    assert "(+1 -1 lines vs previous)" in res
    off(monkeypatch, n)
    res = tools.tool_write(str(p), "a = 1\nb = 4\n")
    assert res.startswith("[wrote") and "lines vs previous" not in res


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
