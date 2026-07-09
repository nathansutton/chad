"""GPU-free end-to-end test of the agent's multi-turn tool loop (plan 056).

The fast unit gate loads no model, so nothing here exercises the *spine* of the product
end to end: render transcript → stream an assistant turn → parse `<tool_call>` →
validate → dispatch a real tool → append the `role:"tool"` result → loop until done. A
regression in that integration (parse-to-dispatch, edit application, loop termination) is
invisible to the per-function unit tests and only surfaces in the private GPU workshop.

This drives a REAL `Agent.run_turn` against the REAL tools with NO weights, NO GPU, and NO
network by swapping the MLX `Engine` for a `ScriptedEngine`: a structural `BaseEngine`
whose `generate` returns pre-authored assistant turns (canned `<tool_call>` blocks)
instead of sampling a model. The tokenizer is a tiny fake — the scripted engine ignores
the rendered prompt, so the render path only needs to produce a length (see
`_FakeTok.apply_chat_template`); no chat template, no download. `mode="auto"` auto-approves
the confirm gate — this test is about the LOOP, not the gate (plan 050 covers the gate).

Mirrors the fake-engine style of test_openai_engine.py; hermetic via `tmp_path`.
"""

import json

from chad.agent import Agent
from chad.base_engine import BaseEngine, GenStats


class _FakeTok:
    """Minimal tokenizer stand-in. `Agent._render` calls only `apply_chat_template`, and
    the scripted engine ignores the returned ids entirely — the loop uses them purely for
    a length (context gauge / compaction threshold). So we return a deterministic,
    comfortably-under-`ctx_limit` id list derived from the transcript size; no real chat
    template, no model files, no network."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                            enable_thinking=False):
        n = sum(len(m.get("content", "")) for m in messages) // 4 + 8
        return list(range(n))

    def decode(self, ids, skip_special_tokens=False):  # unused by the scripted path
        return ""


class ScriptedEngine:
    """A structural `BaseEngine` that replays canned assistant turns (plan 056).

    `generate` ignores the prompt and returns the NEXT string from `script`, honoring
    `OpenAIEngine.generate`'s exact return contract `(text, GenStats)` so `Agent` can't
    tell it apart from a real backend. Stateless, so the warm-prefix / cache-quarantine
    members no-op (like `OpenAIEngine`). If the script runs dry the loop failed to
    terminate — we raise rather than hang, turning a non-terminating loop into a clear
    test failure."""

    def __init__(self, script, model_id="scripted-test", effective_ctx=24000):
        self.script = list(script)
        self._i = 0
        self.model_id = model_id
        self.effective_ctx = effective_ctx
        self.cache_dir = None          # None disables the warm-start prefix path entirely
        self._cached_ids = []          # kept for seam compatibility; never populated
        self.tok = _FakeTok()

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, stop_texts=None,
                 should_stop=None, on_prefill=None, on_prefill_progress=None,
                 stop_condition=None):
        if on_prefill:                 # fire once with no cached prefix, like OpenAIEngine
            on_prefill(len(prompt_ids), 0)
        if self._i >= len(self.script):
            raise AssertionError("scripted engine ran dry — the agent loop did not "
                                 "terminate within the provided turns")
        text = self.script[self._i]
        self._i += 1
        if on_token:
            on_token(text)
        stats = GenStats(prompt_tokens=len(prompt_ids), cached_tokens=0,
                         generated_tokens=max(1, len(text) // 4), approximate=True)
        return text, stats

    # --- stateless seam: no cache to warm, quarantine, or drop ---------------
    def reset(self):
        self._cached_ids = []

    def warm_prefix(self, prefix_ids, should_stop=None):
        return "skip", 0

    def push_cache(self):
        pass

    def pop_cache(self):
        pass


def _tool_call(name, **args):
    """One `<tool_call>` block in the JSON dialect `toolcall_parse` accepts (built with
    `json.dumps` so paths/content are escaped correctly)."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _agent(script, **kw):
    # thinking=False: the scripted turns carry no <think> block, so we skip the
    # template's think handling (and close_unclosed_think) for a clean, literal turn.
    return Agent(ScriptedEngine(script), mode="auto", thinking=False, **kw)


# --- Step 1: the scripted engine structurally satisfies BaseEngine -----------

def test_scripted_engine_satisfies_base_engine_protocol():
    eng = ScriptedEngine(["done"])
    assert isinstance(eng, BaseEngine)


def test_scripted_agent_constructs_without_weights_or_network():
    # Step 2 verify: an Agent builds on the scripted engine + fake tok — no model load.
    agent = _agent(["hi"])
    assert agent.mode == "auto"
    assert agent.engine.model_id == "scripted-test"


def test_template_ids_unwraps_batchencoding():
    # Regression: some HF tokenizers return a dict-like BatchEncoding from
    # apply_chat_template. `list()` of that yields its keys, which the --backend llama
    # path shipped as the prompt (garbage → model degeneration). Coerce to input_ids.
    class _BatchEncodingLike(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    be = _BatchEncodingLike(input_ids=[1, 2, 3], attention_mask=[1, 1, 1])
    assert Agent._template_ids(be) == [1, 2, 3]
    # A plain int list (the MLX path) passes through untouched.
    assert Agent._template_ids([4, 5, 6]) == [4, 5, 6]


# --- Step 3: drive a multi-step task end to end ------------------------------

def test_agent_loop_writes_file_reads_it_back_then_terminates(tmp_path):
    """write → read → done: two real tool dispatches through a real run_turn, a real
    filesystem effect, and clean termination (no spin to max_steps)."""
    target = tmp_path / "note.txt"       # .txt: a doc write, so no verify-before-done nudge
    body = "hello from the scripted loop\n"
    script = [
        _tool_call("write", path=str(target), content=body),
        _tool_call("read", path=str(target)),
        _tool_call("done", summary="wrote and read back the file"),
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("create note.txt")

    # the real `write` tool actually wrote the file to disk
    assert target.exists()
    assert target.read_text() == body
    # the loop fed the tool results back as role:"tool" turns (parse→dispatch→feed-back)
    tool_turns = [m for m in agent.messages if m.get("role") == "tool"]
    assert [m["name"] for m in tool_turns] == ["write", "read"]
    assert tool_turns[0]["content"].startswith("[wrote")
    assert body.strip() in tool_turns[1]["content"]   # `read` observed what `write` wrote
    # the loop terminated on `done` — it did not run out of steps or drain the script
    assert result == "wrote and read back the file"
    assert agent.engine._i == len(script)


def test_agent_loop_terminates_on_a_plain_final_answer(tmp_path):
    """A read-only task that ends with a no-tool-call assistant turn returns that text —
    the other loop-exit path (final answer vs the `done` terminal tool)."""
    target = tmp_path / "data.txt"
    target.write_text("42\n")
    script = [
        _tool_call("read", path=str(target)),
        "The file contains the number 42.",   # no tool call -> final answer, loop ends
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("what's in data.txt?")

    assert result == "The file contains the number 42."
    assert [m["name"] for m in agent.messages if m.get("role") == "tool"] == ["read"]


def test_agent_loop_surfaces_a_real_dispatch_failure(tmp_path):
    """Negative control (plan 056 Step 3 verify): if a dispatch genuinely fails, the loop
    must NOT silently 'succeed'. Pointing `write` at a path under a non-existent file (so
    the parent isn't a directory) makes the real tool raise; the loop feeds the error back
    as the tool result rather than pretending the file was written."""
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("i am a file, not a directory\n")
    doomed = not_a_dir / "child.txt"     # parent is a file -> os.makedirs / open fails
    script = [
        _tool_call("write", path=str(doomed), content="never lands"),
        _tool_call("done", summary="claims success"),
    ]
    agent = _agent(script, max_steps=10)

    agent.run_turn("write a file that can't be written")

    assert not doomed.exists()           # the write really did fail on disk
    write_turn = next(m for m in agent.messages
                      if m.get("role") == "tool" and m["name"] == "write")
    assert write_turn["content"].startswith("[tool error")
    # made_edit never got set, so the loop can't have mistaken the failure for a landed edit
    assert "[wrote" not in write_turn["content"]


class _TokenizingEngine(ScriptedEngine):
    """ScriptedEngine that honors `stop_condition` the way the real engine does: feed
    the scripted text in ~token-sized chunks, consult stop_condition(text_so_far, n)
    after each, and on a hit truncate the turn and set stats.stop_condition_fired —
    the exact contract of engine.generate's decode loop. Lets the run_turn branches
    that react to a mid-generation stop be driven without a model."""

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, stop_texts=None,
                 should_stop=None, on_prefill=None, on_prefill_progress=None,
                 stop_condition=None):
        if self._i >= len(self.script):
            raise AssertionError("scripted engine ran dry — the agent loop did not "
                                 "terminate within the provided turns")
        full = self.script[self._i]
        self._i += 1
        text, n, fired = "", 0, False
        for i in range(0, len(full), 4):        # ~4 chars per "token"
            text += full[i:i + 4]
            n += 1
            if stop_condition is not None and stop_condition(text, n):
                fired = True
                break
        stats = GenStats(prompt_tokens=len(prompt_ids), cached_tokens=0,
                         generated_tokens=n, approximate=True)
        stats.stop_condition_fired = fired
        return text, stats


def test_agent_loop_cuts_off_degenerate_repetition():
    """A step whose output locks into repeating one short string must be cut off early
    (not ground to the token cap), nudged, and the turn must still end with the model's
    NEXT (healthy) answer — the dogfood-trace runaway, replayed without a model."""
    runaway = "The answer starts well " + "`CHAD_NO_TASK`, " * 400   # ~6.4k chars of loop
    script = [runaway, "The flags live in config.py."]
    agent = Agent(_TokenizingEngine(script), mode="auto", thinking=False, max_steps=10)

    result = agent.run_turn("which file centralizes the CHAD_ flags?")

    # the healthy second turn is the final answer — the loop recovered
    assert result == "The flags live in config.py."
    # the degenerate turn was stopped a fraction of the way in, not stored whole
    degen = next(m for m in agent.messages if m.get("role") == "assistant")
    assert len(degen["content"]) < len(runaway) / 2
    # the model was told why before its next step
    assert any("degenerated into repeating" in m.get("content", "")
               for m in agent.messages if m.get("role") == "tool")


# --- Progress-aware step cap: productive turns extend, stalled ones bank a note ------

def test_step_cap_extends_while_turn_lands_verified_changes(tmp_path):
    """A turn that keeps landing AND verifying edits must survive past max_steps (the
    plan-064 trace: a productive plan-implementation turn was force-stopped dead at the
    fixed cap, an edit half-applied). With max_steps=4 this script needs 7 steps — each
    window re-earns its extension with an edit+verify, so the loop reaches `done`."""
    f = tmp_path / "f.py"
    # Distinct args per step — identical repeated calls would (correctly) trip the
    # repeat-loop guard instead of exercising the cap.
    script = []
    for i in range(3):
        # The verify step must be an EXECUTING command (python …) — a display command
        # like `echo` no longer clears unverified_edit (bash_result_verifies).
        script += [_tool_call("write", path=str(f), content=f"x = {i}\n"),
                   _tool_call("bash", command=f"python {f} && echo ok{i}")]
    script.append(_tool_call("done", summary="finished the long task"))
    agent = _agent(script, max_steps=4)

    result = agent.run_turn("keep landing verified changes")

    assert result == "finished the long task"
    assert agent.engine._i == len(script)   # ran past the base cap of 4, to completion
    assert agent.budget_note is None        # clean finish — nothing banked


def test_step_cap_stops_and_banks_note_without_progress(tmp_path):
    """A turn that reaches the cap with no landed+verified change in the window must
    stop (no extension) and bank a progress note — same contract as a governor hard
    stop — so the caller can resume instead of silently dropping the task."""
    target = tmp_path / "data.txt"
    target.write_text("42\n")
    script = [_tool_call("read", path=str(target)),
              _tool_call("read", path=str(target))]
    agent = _agent(script, max_steps=2)

    result = agent.run_turn("read things forever")

    assert "step cap" in result             # explicit stop, not a silent death
    assert agent.budget_note                # note banked for continue/--auto-continue
    assert agent.engine._i == len(script)   # stopped exactly at the cap, no extension


# --- Iter-2 (plan 066): no-empty-diff terminal gates --------------------------------

def test_no_empty_diff_gate_blocks_prose_end_on_action_task():
    """An ACTION task whose model stalls into prose 'final answers' (the NIGHT-7 bail
    signature: django-14007/sphinx-9230 accepted a 'Let me search…' sentence as the
    final answer with an EMPTY diff and 97% of budget unused) must end as a resumable
    hard stop with a progress note — never as a silent success."""
    script = [
        "Let me find where the bug is defined.",   # bail 1 -> nudge
        "Let me search for the relevant code.",    # bail 2 -> nudge (budget exhausted)
        "The fix should go in utils.py.",          # would have been accepted before
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("fix the crash in utils.py")

    assert result.startswith("[stopped:")
    assert "verified change" in result
    assert agent.budget_note                # relaunch seed for --auto-continue


def test_no_empty_diff_gate_blocks_done_with_unverified_edit(tmp_path):
    """`done` after the verify nudges are exhausted, with an edit in tree and no
    successful run since (matplotlib-25332 r3: done at 84s, zero post-edit commands
    succeeded, no guard fired) becomes a resumable hard stop."""
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    script = [
        _tool_call("edit", path=str(f), old="x = 1", new="x = 2"),
        _tool_call("done", summary="changed it"),    # -> verify nudge 1
        _tool_call("done", summary="changed it."),   # -> verify nudge 2
        _tool_call("done", summary="changed it!"),   # nudges exhausted -> gate
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("change x to 2 in m.py")

    assert result.startswith("[stopped:")
    assert agent.budget_note
    assert f.read_text() == "x = 2\n"       # the edit itself stays on disk


def test_prose_answer_still_ends_read_only_turns(tmp_path):
    """Negative control for the gate: an explain-only ask still ends normally on a
    prose answer — the gate must key on action intent, not fire universally."""
    target = tmp_path / "data.txt"
    target.write_text("42\n")
    script = [_tool_call("read", path=str(target)),
              "It contains the number 42."]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("what does data.txt contain?")

    assert result == "It contains the number 42."
    assert agent.budget_note is None


def test_bash_mutation_triggers_syntax_recheck(tmp_path):
    """Iter-2 (plan 066, sphinx-7440): bash can rewrite files (sed -i and friends)
    but used to bypass the edit-tool syntax gate — a file survived 9 blind 'fixes'
    unparseable and nothing said so. A bash step that mutates a file edited this
    turn must get a parse warning appended to its result."""
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    breaker = f"python -c \"open(r'{f}','w').write('def f(:\\n')\""
    script = [
        _tool_call("edit", path=str(f), old="x = 1", new="x = 2"),
        _tool_call("bash", command=breaker),      # bash breaks the watched file
        _tool_call("bash", command=f"python {f}"),  # (fails; keeps turn alive)
        _tool_call("done", summary="attempted"),
    ]
    agent = _agent(script, max_steps=10)
    agent.run_turn("change x to 2 in m.py")

    bash_msgs = [m["content"] for m in agent.messages
                 if m.get("role") == "tool" and m.get("name") == "bash"]
    assert any("no longer parses" in c for c in bash_msgs)


# --- backend-error resilience (plan 068) -------------------------------------
# A transient llama.cpp fault used to escape run_turn and kill the process from
# cli.main, forfeiting the rest of an unattended task's budget: TB2's
# make-mips-interpreter died at 721s of a 1770s budget on a single 500
# ("The model produced output that does not match the expected Content-only format").

class _FlakyEngine(ScriptedEngine):
    """Raises `BackendError(transient=...)` on the first `n_fail` generate calls, then
    replays the script. Records how many times generate was entered."""

    def __init__(self, script, n_fail=1, transient=True, **kw):
        super().__init__(script, **kw)
        self.n_fail = n_fail
        self.transient = transient
        self.calls = 0

    def generate(self, prompt_ids, **kw):
        self.calls += 1
        if self.calls <= self.n_fail:
            from chad.base_engine import BackendError
            raise BackendError("llama-server error: {'code': 500}", transient=self.transient)
        return super().generate(prompt_ids, **kw)


def test_transient_backend_error_is_retried_and_the_turn_completes(monkeypatch):
    monkeypatch.setattr("chad.agent.time.sleep", lambda *_: None)   # no backoff in tests
    eng = _FlakyEngine(["all done"], n_fail=1, transient=True)
    agent = Agent(eng, mode="auto", thinking=False)
    agent.run_turn("do the thing")
    assert eng.calls == 2, "the failed step should have been re-issued exactly once"
    assert agent.messages[-1]["content"] == "all done"


def test_transient_backend_errors_give_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setattr("chad.agent.time.sleep", lambda *_: None)
    import pytest

    from chad.base_engine import BackendError
    eng = _FlakyEngine(["unreachable"], n_fail=99, transient=True)
    agent = Agent(eng, mode="auto", thinking=False)
    with pytest.raises(BackendError):
        agent.run_turn("do the thing")
    # 3 retries + the original attempt: a server that is genuinely down must surface,
    # not silently eat the task's whole budget.
    assert eng.calls == 4


def test_non_transient_backend_error_is_not_retried(monkeypatch):
    monkeypatch.setattr("chad.agent.time.sleep", lambda *_: None)
    import pytest

    from chad.base_engine import BackendError
    eng = _FlakyEngine(["unreachable"], n_fail=99, transient=False)
    agent = Agent(eng, mode="auto", thinking=False)
    with pytest.raises(BackendError):
        agent.run_turn("do the thing")
    assert eng.calls == 1, "a 4xx is the prompt's fault; re-rolling it is wasted budget"


def test_backend_retry_budget_resets_per_turn(monkeypatch):
    monkeypatch.setattr("chad.agent.time.sleep", lambda *_: None)
    eng = _FlakyEngine(["first", "second"], n_fail=1, transient=True)
    agent = Agent(eng, mode="auto", thinking=False)
    agent.run_turn("one")
    eng.n_fail, eng.calls = 3, 0      # a fresh transient fault on the next turn
    eng._i = 1                        # replay from "second"
    agent.run_turn("two")
    assert eng.calls == 4             # 3 failures re-rolled, 4th succeeds
