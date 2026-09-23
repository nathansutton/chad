"""GPU-free end-to-end test of the agent's multi-turn tool loop.

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
`_FakeTok.apply_chat_template`); no chat template, no download. `mode="yolo"` auto-approves
the confirm gate — this test is about the LOOP, not the gate (covers the gate).

Mirrors the fake-engine style of test_completion_engine.py; hermetic via `tmp_path`.
"""

import json
import os
import shlex
import sys

import pytest

from chad import guardrails, tools
from chad.agent import THINK_CEILING_DEFAULT, Agent, reject_escalation
from chad.base_engine import BaseEngine, GenStats

# The interpreter running the tests, not whatever `python` PATH happens to hold.
# A bare `python` does not exist on a stock macOS (only `python3`), so these bash
# steps exited 127 and the scripted engine ran dry — a launcher-dependent failure
# that looked like a loop bug.
PY = shlex.quote(sys.executable)


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
    """A structural `BaseEngine` that replays canned assistant turns.

    `generate` ignores the prompt and returns the NEXT string from `script`, honoring
    `CompletionEngine.generate`'s exact return contract `(text, GenStats)` so `Agent` can't
    tell it apart from a real backend. Stateless, so `warm_prefix` no-ops (like
    `CompletionEngine`). If the script runs dry the loop failed to
    terminate — we raise rather than hang, turning a non-terminating loop into a clear
    test failure."""

    def __init__(self, script, model_id="scripted-test", effective_ctx=24000):
        self.script = list(script)
        self._i = 0
        self.model_id = model_id
        self.effective_ctx = effective_ctx
        self.cache_dir = None          # None disables the warm-start prefix path entirely
        self._cached_ids = []          # kept for seam compatibility; never populated
        self.reasoning_effort_default = None
        self.tok = _FakeTok()

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, stop_texts=None,
                 should_stop=None, on_prefill=None, on_prefill_progress=None,
                 stop_condition=None, think_ceiling=None):
        if on_prefill:                 # fire once with no cached prefix, like CompletionEngine
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

    # --- stateless seam: no cache to warm or drop ---------------
    def reset(self):
        self._cached_ids = []

    def warm_prefix(self, prefix_ids, should_stop=None, head_ids=None):
        return "skip", 0


def _tool_call(name, **args):
    """One `<tool_call>` block in the JSON dialect `toolcall_parse` accepts (built with
    `json.dumps` so paths/content are escaped correctly)."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _agent(script, **kw):
    # thinking=False: the scripted turns carry no <think> block, so we skip the
    # template's think handling (and close_unclosed_think) for a clean, literal turn.
    return Agent(ScriptedEngine(script), mode="yolo", thinking=False, **kw)


# --- Step 1: the scripted engine structurally satisfies BaseEngine -----------

def test_scripted_engine_satisfies_base_engine_protocol():
    eng = ScriptedEngine(["done"])
    assert isinstance(eng, BaseEngine)


def test_scripted_agent_constructs_without_weights_or_network():
    # Step 2 verify: an Agent builds on the scripted engine + fake tok — no model load.
    agent = _agent(["hi"])
    assert agent.mode == "yolo"
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

def test_agent_loop_writes_file_reads_it_back_then_terminates(tmp_path, monkeypatch):
    """write → bash → done: two real tool dispatches through a real run_turn, a real
    filesystem effect, and clean termination (no spin to max_steps)."""
    monkeypatch.chdir(tmp_path)          # tmp_path is the WORKSPACE, as in a real run
    target = tmp_path / "note.txt"       # .txt: a doc write, so no verify-before-done nudge
    body = "hello from the scripted loop\n"
    script = [
        _tool_call("write", path=str(target), content=body),
        _tool_call("bash", command="cat " + str(target)),
        _tool_call("done", summary="wrote and read back the file"),
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("create note.txt")

    # the real `write` tool actually wrote the file to disk
    assert target.exists()
    assert target.read_text() == body
    # the loop fed the tool results back as role:"tool" turns (parse→dispatch→feed-back)
    tool_turns = [m for m in agent.messages if m.get("role") == "tool"]
    assert [m["name"] for m in tool_turns] == ["write", "bash"]
    assert tool_turns[0]["content"].startswith("[wrote")
    assert body.strip() in tool_turns[1]["content"]   # bash observed what `write` wrote
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
    # `read` is not an exposed tool; tools.alias_to_bash rewrites it to the `bash` that
    # reads the file, so the recorded tool message is the bash it became.
    assert [m["name"] for m in agent.messages if m.get("role") == "tool"] == ["bash"]
    assert "42" in agent.messages[-2]["content"]


def test_agent_loop_surfaces_a_real_dispatch_failure(tmp_path, monkeypatch):
    """Negative control (verify): if a dispatch genuinely fails, the loop
    must NOT silently 'succeed'. Pointing `write` at a path under a non-existent file (so
    the parent isn't a directory) makes the real tool raise; the loop feeds the error back
    as the tool result rather than pretending the file was written. (The churn
    handoff would rightly bounce the empty-diff done first — disabled here; this test
    is about dispatch, and the handoff has its own coverage in test_done_audit.py.)"""
    monkeypatch.chdir(tmp_path)          # tmp_path is the WORKSPACE, as in a real run
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
                 stop_condition=None, think_ceiling=None):
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
    agent = Agent(_TokenizingEngine(script), mode="yolo", thinking=False, max_steps=10)

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

def test_step_cap_extends_while_turn_lands_verified_changes(tmp_path, monkeypatch):
    """A turn that keeps landing AND verifying edits must survive past max_steps (a
    real trace: a productive plan-implementation turn was force-stopped dead at the
    fixed cap, an edit half-applied). With max_steps=4 this script needs 7 steps — each
    window re-earns its extension with an edit+verify, so the loop reaches `done`."""
    # Orthogonal to the deliverable recheck (it would add a step and skew the cap
    # accounting this test pins); disable that lever here.
    monkeypatch.chdir(tmp_path)          # tmp_path is the WORKSPACE, as in a real run
    f = tmp_path / "f.py"
    # Distinct args per step — identical repeated calls would (correctly) trip the
    # repeat-loop guard instead of exercising the cap.
    script = []
    for i in range(3):
        # The verify step must be an EXECUTING command (python …) — a display command
        # like `echo` no longer clears unverified_edit (bash_result_verifies).
        script += [_tool_call("write", path=str(f), content=f"x = {i}\n"),
                   _tool_call("bash", command=f"{PY} {f} && echo ok{i}")]
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


# --- Iter-2: no-empty-diff terminal gates --------------------------------

def test_no_empty_diff_gate_blocks_prose_end_on_action_task(monkeypatch):
    """An ACTION task whose model stalls into prose 'final answers' (the measured bail
    signature: a 'Let me search…' sentence accepted as the final answer with an EMPTY
    diff and 97% of budget unused) must end as a resumable
    hard stop with a progress note — never as a silent success. (This test is about
    the GATE; the churn handoff would insert one audit bounce first — its own
    coverage lives in test_done_audit.py.)"""
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


def test_no_empty_diff_gate_blocks_done_with_unverified_edit(tmp_path, monkeypatch):
    """`done` after the verify nudges are exhausted, with an edit in tree and no
    successful run since (measured: done at 84s, zero post-edit commands succeeded,
    no guard fired) becomes a resumable hard stop. (Gate-focused: the
    churn handoff — one audit bounce before this stop — is disabled here and
    covered in test_done_audit.py.)"""
    monkeypatch.chdir(tmp_path)          # tmp_path is the WORKSPACE, as in a real run
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


def test_bash_mutation_triggers_syntax_recheck(tmp_path, monkeypatch):
    """Iter-2: bash can rewrite files (sed -i and friends)
    but used to bypass the edit-tool syntax gate — a file survived 9 blind 'fixes'
    unparseable and nothing said so. A bash step that mutates a file edited this
    turn must get a parse warning appended to its result."""
    # Not about the deliverable recheck (it would defer done here); disable it.
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    breaker = f"{PY} -c \"open(r'{f}','w').write('def f(:\\n')\""
    script = [
        _tool_call("edit", path=str(f), old="x = 1", new="x = 2"),
        _tool_call("bash", command=breaker),      # bash breaks the watched file
        _tool_call("bash", command=f"{PY} {f}"),  # (fails; keeps turn alive)
        _tool_call("done", summary="attempted"),
    ]
    agent = _agent(script, max_steps=10)
    agent.run_turn("change x to 2 in m.py")

    bash_msgs = [m["content"] for m in agent.messages
                 if m.get("role") == "tool" and m.get("name") == "bash"]
    assert any("no longer parses" in c for c in bash_msgs)


# --- backend-error resilience -------------------------------------
# A transient llama.cpp fault used to escape run_turn and kill the process from
# cli.main, forfeiting the rest of an unattended task's budget — a measured run died
# at 721s of a 1770s budget on a single 500 ("The model produced output that does not
# match the expected Content-only format").

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


# --- Mid-run steering (improve 01) -------------------------------------------
# User text typed while a turn runs is drained between steps and injected as a
# synthetic `role:"tool", name:"steer"` message — a pure append, so the warm KV
# prefix stays valid (the whole point vs interrupt + re-prefill). These drive the
# REAL run_turn drain point with a scripted engine that "types" a steer after the
# first assistant turn streams, exactly like the TUI's worker-thread wiring.

class _SteerAfterFirstTurn(ScriptedEngine):
    """Enqueues `steer_text` into `steer_queue` right after the FIRST scripted turn
    returns — emulating a user typing while step 0's tool batch executes."""

    def __init__(self, script, steer_queue, steer_text, **kw):
        super().__init__(script, **kw)
        self._steer_queue = steer_queue
        self._steer_text = steer_text

    def generate(self, *a, **kw):
        out = super().generate(*a, **kw)
        if self._i == 1:
            self._steer_queue.append(self._steer_text)
        return out


def test_steering_injects_between_steps_and_run_continues(tmp_path, monkeypatch):
    """The steer lands in `messages` after step 0's tool result and before step 1's
    assistant turn, framed as an overriding tool-role message; the run continues to
    `done` (interrupted stays False)."""
    monkeypatch.chdir(tmp_path)          # tmp_path is the WORKSPACE, as in a real run
    target = tmp_path / "note.txt"
    steer_text = "actually, stop — the OTHER file is the target"
    script = [
        _tool_call("write", path=str(target), content="v1\n"),
        _tool_call("read", path=str(target)),
        _tool_call("done", summary="finished"),
    ]
    steers = []
    eng = _SteerAfterFirstTurn(script, steers, steer_text)

    def drain():
        out, steers[:] = list(steers), []
        return out

    agent = Agent(eng, mode="yolo", thinking=False, drain_steering=drain, max_steps=10)
    result = agent.run_turn("create note.txt")

    assert result == "finished"
    assert agent.interrupted is False
    idx = [i for i, m in enumerate(agent.messages)
           if m.get("role") == "tool" and m.get("name") == "steer"]
    assert len(idx) == 1, "the steer must be injected exactly once"
    i = idx[0]
    # after step 0's tool result...
    prev = agent.messages[i - 1]
    assert (prev.get("role"), prev.get("name")) == ("tool", "write")
    # ...and before step 1's assistant turn (between steps, never inside a tool batch)
    assert agent.messages[i + 1].get("role") == "assistant"
    # framed as user steering that overrides the original ask, with the text verbatim
    assert agent.messages[i]["content"].startswith("[user steering — ")
    assert agent.messages[i]["content"].endswith(steer_text)


def test_no_drain_hook_means_no_injection(tmp_path, monkeypatch):
    """drain_steering=None (headless / bench / sub-agent) keeps today's transcript
    byte-identical — no steer messages, no behavior change (zero benchmark risk)."""
    target = tmp_path / "note.txt"
    script = [
        _tool_call("write", path=str(target), content="v1\n"),
        _tool_call("done", summary="finished"),
    ]
    agent = _agent(script, max_steps=10)
    agent.run_turn("create note.txt")
    assert not any(m.get("name") == "steer" for m in agent.messages)


class _SeqTok(_FakeTok):
    """Content-faithful fake tokenizer: the render is a per-message concatenation of
    the role+content bytes, so appending a message EXTENDS the render while mutating
    or reordering an earlier one changes its tokens. This is exactly the property
    engine.py's prefix diff relies on to prefill only the appended tail."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                            enable_thinking=False):
        ids = []
        for m in messages:
            ids.extend(ord(c) for c in m.get("role", "") + ":" + str(m.get("content", "")))
            ids.append(0)  # message separator
        return ids


def test_steering_keeps_the_render_prefix_stable(tmp_path, monkeypatch):
    """The cache-safety claim itself: every render the engine sees — including the one
    right after the injection — is a pure EXTENSION of the previous render (common
    prefix == the whole previous prompt). A steer that rewrote or reordered history
    would break this and force a full re-prefill on Ornith's non-trimmable cache."""
    target = tmp_path / "note.txt"
    script = [
        _tool_call("write", path=str(target), content="v1\n"),
        _tool_call("read", path=str(target)),
        _tool_call("done", summary="finished"),
    ]
    steers = []
    eng = _SteerAfterFirstTurn(script, steers, "use two-space indent everywhere")
    eng.tok = _SeqTok()
    renders = []
    real_generate = eng.generate

    def generate(prompt_ids, **kw):
        renders.append(list(prompt_ids))
        return real_generate(prompt_ids, **kw)

    eng.generate = generate

    def drain():
        out, steers[:] = list(steers), []
        return out

    # _SeqTok yields one id per CHARACTER, so the ~44k-char system prompt alone would
    # cross the default 24k compaction threshold; raise it — this test is about the
    # injection's prefix purity, not compaction (test_compaction.py owns that).
    agent = Agent(eng, mode="yolo", thinking=False, drain_steering=drain,
                  max_steps=10, ctx_limit=200_000)
    agent.run_turn("create note.txt")

    assert len(renders) == 3          # one per scripted turn; steer forced no re-roll
    for prev, cur in zip(renders, renders[1:]):
        assert cur[: len(prev)] == prev, (
            "a render stopped being a pure extension of its predecessor — the steer "
            "injection invalidated the warm KV prefix")


# --- No-think escalation ------------------------------------------

class _ThinkFlagTok(_FakeTok):
    """Records the `enable_thinking` passed to each render, so a test can see WHICH steps
    the loop chose to render with <think> disabled (the no-think escalation).
    `warm_prefix` is gated behind cache_dir (None here), so every recorded flag is a step
    render — nothing else calls apply_chat_template on the scripted path."""

    def __init__(self):
        self.flags = []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                            enable_thinking=False):
        self.flags.append(enable_thinking)
        return super().apply_chat_template(messages, tools, add_generation_prompt,
                                           enable_thinking)


class _CappedEngine(ScriptedEngine):
    """Every scripted turn reports hit_cap (generated_tokens == max_tokens), so a no-tool
    turn reads as a token-cap truncation — the think-spiral stall that arms no-think
    escalation after two in a row."""

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, **kw):
        text, stats = super().generate(prompt_ids, max_tokens, on_token, **kw)
        stats.generated_tokens = max_tokens
        return text, stats


def _run_escalation(monkeypatch, *, ablated: bool):
    # Arm the close-and-continue ceiling so escalation is active at all (it is gated on the
    # ceiling being set, so default chad is unaffected).
    monkeypatch.setenv("CHAD_THINK_CEILING", "3000")
    if ablated:
        monkeypatch.setenv("CHAD_DISABLE", "no_think_escalation")
    else:
        monkeypatch.delenv("CHAD_DISABLE", raising=False)
    # Capped no-tool-call stalls: the model just "thinks" and never acts (the spiral).
    eng = _CappedEngine(["<think>stalling</think>"] * 8)
    tok = _ThinkFlagTok()
    eng.tok = tok
    # thinking=True so there IS a <think> to disable; max_gen_tokens tiny so the forced
    # generated_tokens == max_tokens trips hit_cap.
    agent = Agent(eng, mode="yolo", thinking=True, max_gen_tokens=64)
    agent.run_turn("change the config value")   # an action task, not read-only
    return tok.flags


class _CeilingEngine(ScriptedEngine):
    """Records the think ceiling each step's generate was handed."""

    def __init__(self, script):
        super().__init__(script)
        self.ceilings = []

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, **kw):
        self.ceilings.append(kw.get("think_ceiling"))
        return super().generate(prompt_ids, max_tokens, on_token, **kw)


@pytest.mark.parametrize("env, want", [(None, THINK_CEILING_DEFAULT), ("0", None), ("900", 900)])
def test_the_think_ceiling_is_on_by_default_and_zero_turns_it_off(monkeypatch, env, want):
    if env is None:
        monkeypatch.delenv("CHAD_THINK_CEILING", raising=False)
    else:
        monkeypatch.setenv("CHAD_THINK_CEILING", env)
    eng = _CeilingEngine(["<think>ok</think>done"])
    eng.tok = _ThinkFlagTok()
    Agent(eng, mode="yolo", thinking=True).run_turn("what is in this repo?")
    assert eng.ceilings == [want]


class _BigThinkEngine(ScriptedEngine):
    """Every scripted turn is (almost) entirely a `<think>` block and reports a large
    `generated_tokens`, so the turn's CUMULATIVE think spend crosses the turn-level budget
    thresholds within a couple of steps rather than needing thousands of tiny ones."""

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, **kw):
        text, stats = super().generate(prompt_ids, max_tokens, on_token, **kw)
        stats.generated_tokens = 15000
        return text, stats


def _run_turn_think_budget(monkeypatch, *, ablated: bool, script=None,
                           turn_budget_s=1e9):
    if ablated:
        monkeypatch.setenv("CHAD_DISABLE", "turn_think_budget")
    else:
        monkeypatch.delenv("CHAD_DISABLE", raising=False)
    stall = "<think>" + "reasoning " * 50 + "</think>"
    eng = _BigThinkEngine(script if script is not None else [stall] * 6)
    tok = _ThinkFlagTok()
    eng.tok = tok
    # A configured wall budget is required to arm the mechanism at all (like
    # wrapup_window); huge so the UNRELATED governor/wrapup checks never fire and
    # muddy the transcript this test inspects.
    agent = Agent(eng, mode="yolo", thinking=True, max_gen_tokens=20000,
                  turn_budget_s=turn_budget_s)
    agent.run_turn("change the config value")   # an action task, not read-only
    return tok.flags


def _run_capped_think(monkeypatch, *, ablated: bool):
    """Drive a turn whose every generation ends INSIDE <think> at the raw token cap:
    no closing tag, and no stop CONDITION fired (the raw cap is not one). Faithful to
    the real generation, which never emits the opening tag either — the chat template
    opens the block, so the model's output is bare reasoning text. `max_gen_tokens`
    matches what the engine reports so each step registers as a cap hit."""
    if ablated:
        monkeypatch.setenv("CHAD_DISABLE", "capped_think_credit")
    else:
        monkeypatch.delenv("CHAD_DISABLE", raising=False)
    truncated = "reasoning " * 50          # opened by the template, never closed
    eng = _BigThinkEngine([truncated] * 6)
    tok = _ThinkFlagTok()
    eng.tok = tok
    # think_ceiling pinned OFF so this stays a clean A/B of `capped_think_credit`.
    # The ceiling is defaulted ON in production, and arming it also arms
    # `no_think_escalation` (which flips the very same thinking flag these arms read,
    # after two capped stalls) — leaving it at the default would let that lever, not
    # the one under test, decide the OFF arm's result.
    agent = Agent(eng, mode="yolo", thinking=True, max_gen_tokens=15000,
                  turn_budget_s=1e9, think_ceiling=0)
    agent.run_turn("change the config value")
    return tok.flags



def test_final_plan_update_paired_with_done_is_not_dropped(tmp_path, monkeypatch):
    """`write_todos` + `done` in ONE step: the terminal short-circuit returns before the
    execute loop, so the model's last plan update — the step where it marks everything
    `[x]` — used to be silently discarded. It is the model's habitual closing move, so
    the recorded plan ended every turn a step stale."""
    from chad import tools

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "note.txt"
    script = [
        _tool_call("write_todos", todos=[{"content": "Write the note", "status": "in_progress"},
                                     {"content": "Read it back", "status": "pending"}]),
        _tool_call("write", path=str(target), content="hi\n"),
        _tool_call("bash", command="cat " + str(target)),
        # The closing step: plan update and done together.
        _tool_call("write_todos", todos=[{"content": "Write the note", "status": "completed"},
                                     {"content": "Read it back", "status": "completed"}])
        + "\n" + _tool_call("done", summary="wrote and read back the note"),
    ]
    agent = _agent(script)
    agent.run_turn("write a note and read it back")

    assert [t["status"] for t in tools._TODOS] == ["completed", "completed"], tools._TODOS
    # The result is in the transcript too, so a `done` that gets rejected replays the
    # turn with the model's own plan visible rather than a hole where the call was.
    plan_results = [m for m in agent.messages
                    if m.get("role") == "tool" and m.get("name") == "write_todos"]
    assert len(plan_results) == 2, [m.get("content") for m in plan_results]
    assert "[x] Read it back" in plan_results[-1]["content"]


def test_edit_paired_with_done_is_applied(tmp_path, monkeypatch):
    """`edit` + `done` in ONE step: the edit runs before the done-gates judge the turn.
    The terminal short-circuit used to drop every call but a plan update, so the edit
    never landed and the model was told it had not done anything."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    script = [
        _tool_call("edit", path=str(f), old="x = 1", new="x = 2")
        + "\n" + _tool_call("done", summary="changed it"),       # -> verify question
        _tool_call("bash", command=f"{PY} {f}"),
        _tool_call("done", summary="changed x to 2 and ran it"),
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("change x to 2 in m.py")

    assert f.read_text() == "x = 2\n"
    tool_turns = [m for m in agent.messages if m.get("role") == "tool"]
    assert [m["name"] for m in tool_turns[:2]] == ["edit", "done"]
    assert tool_turns[0]["content"].startswith("[edited")
    assert "have not run anything" in tool_turns[1]["content"]   # judged the landed edit
    assert result == "changed x to 2 and ran it"
    assert agent.engine._i == len(script)


def test_done_with_an_open_todo_is_questioned_once_then_accepted(tmp_path, monkeypatch):
    """The plan the model wrote this turn holds up `done` for exactly one question.

    Measured on a full TB2.1 run, this is a nudge and not a gate on purpose: of the 75
    turns that reached `done`, only 6 still had an open item and those 6 scored as well
    as the rest, while 10 of the 12 turns that called `done` and scored zero had every
    box ticked. One question is affordable; refusing a healthy run is not.
    """
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "data.txt"
    target.write_text("42\n")
    tools.clear_todos()
    script = [
        _tool_call("write_todos", todos=[{"content": "look at the file", "status": "in_progress"},
                                     {"content": "report the number", "status": "pending"}]),
        _tool_call("bash", command=f"cat {target}"),
        _tool_call("done", summary="read it"),                       # -> questioned
        _tool_call("write_todos", todos=[{"content": "look at the file", "status": "completed"},
                                     {"content": "report the number", "status": "completed"}]),
        _tool_call("done", summary="read it, the number is 42"),     # -> accepted
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn("what does data.txt contain?")

    assert result == "read it, the number is 42"
    nudges = [m["content"] for m in agent.messages
              if m.get("role") == "tool" and m.get("name") == "done"]
    assert len(nudges) == 1, "the open-todo question must fire exactly once"
    assert "report the number" in nudges[0]     # names the item that is still open
    assert agent.engine._i == len(script)
    tools.clear_todos()


def test_done_is_not_questioned_when_every_todo_is_ticked(tmp_path, monkeypatch):
    """Negative control: a finished plan must not cost a step."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "data.txt"
    target.write_text("42\n")
    tools.clear_todos()
    script = [
        _tool_call("write_todos", todos=[{"content": "look at the file", "status": "completed"}]),
        _tool_call("bash", command=f"cat {target}"),
        _tool_call("done", summary="read it"),
    ]
    agent = _agent(script, max_steps=10)

    assert agent.run_turn("what does data.txt contain?") == "read it"
    assert not [m for m in agent.messages
                if m.get("role") == "tool" and m.get("name") == "done"]
    tools.clear_todos()


def test_a_stale_plan_from_an_earlier_turn_does_not_ambush_done(tmp_path, monkeypatch):
    """A plan spans turns by design; only the plan written THIS turn gates this turn."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "data.txt"
    target.write_text("42\n")
    agent = _agent([_tool_call("bash", command=f"cat {target}"),
                    _tool_call("done", summary="unrelated task finished")], max_steps=10)
    # Written after construction: an earlier turn of THIS session, not a leftover from a
    # previous one (which a new Agent drops — see the next test).
    tools.tool_write_todos([{"content": "something left over from a previous request",
                             "status": "pending"}])

    assert agent.run_turn("what does data.txt contain?") == "unrelated task finished"
    assert not [m for m in agent.messages
                if m.get("role") == "tool" and m.get("name") == "done"]
    tools.clear_todos()


def test_a_new_agent_starts_with_an_empty_todo_list(tmp_path, monkeypatch):
    """The todo list is module state that outlives a turn, so a new session would
    otherwise open with the previous one's plan pinned and gate `done` on it."""
    monkeypatch.chdir(tmp_path)
    tools.tool_write_todos([{"content": "left over", "status": "in_progress"}])
    assert tools.unfinished_todos() == ["left over"]
    _agent([_tool_call("done", summary="nothing to do")])
    assert tools.unfinished_todos() == []


# --- run_turn exit branches: each ends the turn with a result its caller reads -------

class _InterruptingEngine(ScriptedEngine):
    """Raises the user's stop flag while producing scripted turn `stop_on` (1-based) —
    a ctrl-c landing mid-generation. The agent's `should_stop` reads `stopped`."""

    def __init__(self, script, stop_on):
        super().__init__(script)
        self.stop_on = stop_on
        self.stopped = False

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, should_stop=None, **kw):
        text, stats = super().generate(prompt_ids, max_tokens, on_token,
                                       should_stop=should_stop, **kw)
        if should_stop is not None and self._i == self.stop_on:
            self.stopped = True
        return text, stats


def test_interrupt_ends_turn_and_marks_agent():
    """ctrl-c mid-generation ends the turn as `[interrupted]`, and the partial turn is
    stored with its think block closed: an unclosed one re-renders as a divergent prefix
    and forces a full re-prefill on the next turn."""
    script = ["look before acting\n</think>\n\n" + _tool_call("bash", command="echo step1"),
              "still reasoning about the"]   # the template opened <think>; never closed
    eng = _InterruptingEngine(script, stop_on=2)
    agent = Agent(eng, mode="yolo", thinking=True, should_stop=lambda: eng.stopped)

    result = agent.run_turn("fix the crash in utils.py")

    assert result == "[interrupted]"
    assert agent.interrupted is True
    assert any(m.get("role") == "tool" and m.get("name") == "bash" for m in agent.messages)
    last = agent.messages[-1]
    assert last["role"] == "assistant"
    assert last["content"].startswith("still reasoning")
    assert last["content"].endswith("</think>")
    # The cut step never closed its think, so it is reasoning end to end — the step a
    # wall clock interrupts is the runaway, and it must not read as zero thinking.
    cut = max(1, len(script[1]) // 4)
    assert agent.think_tokens > cut and agent.gen_tokens > agent.think_tokens


class _StreamingEngine(ScriptedEngine):
    """Streams each turn one character per token and, at `peek_at` tokens, reads the
    trajectory file back — what a harness tailing a live trial would see."""

    def __init__(self, script, path, peek_at):
        super().__init__(script)
        self.path, self.peek_at, self.seen = path, peek_at, None

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, **kw):
        text = self.script[self._i]
        self._i += 1
        for n, ch in enumerate(text, 1):
            on_token(ch)
            if n == self.peek_at and self.seen is None:
                with open(self.path, encoding="utf-8") as f:
                    self.seen = json.load(f)
        return text, GenStats(prompt_tokens=len(prompt_ids), generated_tokens=len(text))


def test_trajectory_shows_the_step_in_flight_then_the_real_one(tmp_path):
    """A think that never ends leaves no message, so the trajectory carries the partial
    text while it is being generated, and swaps in the real step once it lands."""
    from chad import atif
    path = str(tmp_path / "trial.json")
    atif.start(path)
    try:
        think = "trace it by hand " * 20                      # > PENDING_EVERY characters
        eng = _StreamingEngine([think + "</think>\n\nall done"], path, atif.PENDING_EVERY)
        Agent(eng, mode="yolo", thinking=True).run_turn("solve it", stream=False)
    finally:
        atif._reset_for_tests()

    live = eng.seen["steps"][-1]
    assert live["extra"] == {"in_flight": True, "generated_tokens": atif.PENDING_EVERY}
    assert live["reasoning_content"] == think[:atif.PENDING_EVERY] and live["message"] == ""

    with open(path, encoding="utf-8") as f:
        final = json.load(f)
    assert all(not (s.get("extra") or {}).get("in_flight") for s in final["steps"])
    done = final["steps"][-1]
    assert done["metrics"]["extra"]["finish"] == "eos"
    assert done["metrics"]["extra"]["think_tokens"] > 0
    assert final["final_metrics"]["extra"]["total_think_tokens"] == \
        done["metrics"]["extra"]["think_tokens"]


def test_interrupt_while_a_tool_call_is_generated_does_not_dispatch_it(tmp_path):
    """The stop check sits between generation and dispatch: a write the user interrupted
    must not land."""
    target = tmp_path / "f.py"
    eng = _InterruptingEngine([_tool_call("write", path=str(target), content="x = 1\n")],
                              stop_on=1)
    agent = Agent(eng, mode="yolo", thinking=False, should_stop=lambda: eng.stopped)

    assert agent.run_turn("create f.py") == "[interrupted]"
    assert not target.exists()


def test_hard_governor_returns_budget_sentinel(monkeypatch):
    """A turn that spends its token budget with no landed+verified change gets exactly
    one soft nudge, then ends with the `[budget]` result and a banked progress note."""
    monkeypatch.delenv("CHAD_NO_GOVERNOR", raising=False)
    script = [_tool_call("bash", command=f"echo probe{i}") for i in range(4)]
    agent = _agent(script, turn_budget_tokens=10)   # the first prompt alone overshoots it

    result = agent.run_turn("fix the crash in utils.py")

    assert result.startswith(guardrails.BUDGET_SENTINEL)
    assert agent.budget_note and result.endswith(agent.budget_note)
    nudges = [m for m in agent.messages if m.get("content") == guardrails.GOVERNOR_SOFT_NUDGE]
    assert len(nudges) == 1


def test_loop_abort_returns_stuck_message():
    """The same call set five times: nudged on the 3rd and 4th, aborted on the 5th."""
    script = [_tool_call("bash", command="echo same")] * 5
    agent = _agent(script)

    result = agent.run_turn("fix the crash in utils.py")

    assert "stuck in a loop" in result
    nudges = [m for m in agent.messages if "[loop detected" in m.get("content", "")]
    assert len(nudges) == 2
    assert agent.engine._i == len(script)


def test_repeated_invalid_call_escalates(tmp_path):
    """The same schema-invalid call twice: the first rejection is the plain repair
    message, the second adds the stop-repeating escalation."""
    bad = _tool_call("edit", path=str(tmp_path / "x.py"))   # no `old` / `new`
    agent = _agent([bad, bad, "x.py does not exist yet."])

    agent.run_turn("what does x.py contain?")

    rejections = [m["content"] for m in agent.messages
                  if m.get("role") == "tool" and m.get("name") == "edit"]
    assert len(rejections) == 2
    assert reject_escalation("edit") not in rejections[0]
    assert rejections[1].endswith(reject_escalation("edit"))


# --- The workspace boundary: write/edit outside cwd reach a human in every mode ----

def _recorder():
    """A confirm callback that records what it was asked about and answers `answer`."""
    seen = []

    def make(answer):
        def confirm(name, args):
            seen.append((name, args.get("path")))
            return answer
        return confirm
    return seen, make


def test_yolo_write_inside_the_workspace_is_not_questioned(tmp_path, monkeypatch):
    """The boundary must not cost yolo its whole point: an ordinary in-workspace write
    still lands with nobody asked."""
    monkeypatch.chdir(tmp_path)
    seen, make = _recorder()
    target = tmp_path / "inside.txt"
    agent = _agent([_tool_call("write", path=str(target), content="in\n"),
                    _tool_call("done", summary="wrote inside.txt")],
                   confirm=make(True))

    agent.run_turn("create inside.txt")

    assert seen == []
    assert target.read_text() == "in\n"


def test_yolo_write_outside_the_workspace_still_asks(tmp_path, monkeypatch):
    """write/edit run outside the bash seatbelt, so a path that resolves out of the
    working directory is escalated to the human even in yolo."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    seen, make = _recorder()
    outside = tmp_path / "outside.txt"
    agent = _agent([_tool_call("write", path=str(outside), content="out\n"),
                    _tool_call("done", summary="wrote outside.txt")],
                   confirm=make(True))

    agent.run_turn("create outside.txt")

    assert seen == [("write", str(outside))]     # asked once, about that path
    assert outside.read_text() == "out\n"        # approved -> it lands


def test_a_declined_outside_write_does_not_land(tmp_path, monkeypatch):
    """Declining is a plain human "no": the file is untouched and the model is told so."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    seen, make = _recorder()
    outside = tmp_path / "outside.txt"
    agent = _agent([_tool_call("write", path=str(outside), content="out\n"),
                    _tool_call("done", summary="claims success"),
                    _tool_call("done", summary="claims success."),
                    _tool_call("done", summary="claims success!")],
                   confirm=make(False))

    agent.run_turn("create outside.txt")

    assert seen == [("write", str(outside))]
    assert not outside.exists()
    results = [m["content"] for m in agent.messages
               if m.get("role") == "tool" and m.get("name") == "write"]
    assert results == ["[denied by user]"]


def test_headless_outside_write_is_blocked_with_a_reason(tmp_path, monkeypatch):
    """No TTY and no callback: there is nobody to escalate to, so the write is blocked
    and the model gets the resolved path and the reason instead of a bare refusal."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    outside = tmp_path / "outside.txt"
    agent = _agent([_tool_call("write", path=str(outside), content="out\n"),
                    _tool_call("done", summary="claims success"),
                    _tool_call("done", summary="claims success."),
                    _tool_call("done", summary="claims success!")],
                   is_tty=lambda: False)

    agent.run_turn("create outside.txt")

    assert not outside.exists()
    results = [m["content"] for m in agent.messages
               if m.get("role") == "tool" and m.get("name") == "write"]
    assert len(results) == 1
    assert results[0].startswith("[blocked: write outside the workspace")
    assert str(os.path.realpath(outside)) in results[0]


def test_a_symlink_out_of_the_workspace_is_caught(tmp_path, monkeypatch):
    """The check is on the REAL path: an in-workspace name that links out still asks."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    (tmp_path / "target.txt").write_text("old\n")
    os.symlink(tmp_path / "target.txt", ws / "innocent.txt")
    seen, make = _recorder()
    agent = _agent([_tool_call("edit", path="innocent.txt", old="old", new="new"),
                    _tool_call("done", summary="edited innocent.txt")],
                   confirm=make(True))

    agent.run_turn("change old to new in innocent.txt")

    assert seen == [("edit", "innocent.txt")]
    assert (tmp_path / "target.txt").read_text() == "new\n"


def test_reasoning_effort_default_comes_from_the_engine(monkeypatch):
    """A loader can carry a reasoning-effort default for a template whose own default
    it disagrees with (the Prism pack's says xhigh; think-decode is two thirds of wall).
    Without one the template's default stands (nothing is passed, so templates that do
    not take the argument render as before); the user's CHAD_REASONING_EFFORT wins."""
    seen = []

    class _Tok(_FakeTok):
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                                enable_thinking=False, **kw):
            seen.append(dict(kw))
            return super().apply_chat_template(messages, tools, add_generation_prompt,
                                               enable_thinking)

    def run(default=None):
        eng = ScriptedEngine(["All done."])
        eng.tok = _Tok()
        if default:
            eng.reasoning_effort_default = default
        seen.clear()
        Agent(eng, mode="yolo", thinking=False).run_turn("hi")
        rendered = [kw for kw in seen if "tokenize" not in kw]   # not the split probe
        assert rendered
        return rendered

    monkeypatch.delenv("CHAD_REASONING_EFFORT", raising=False)
    assert all("reasoning_effort" not in kw for kw in run())
    assert all(kw.get("reasoning_effort") == "medium" for kw in run("medium"))
    monkeypatch.setenv("CHAD_REASONING_EFFORT", "low")
    assert all(kw.get("reasoning_effort") == "low" for kw in run("medium"))
