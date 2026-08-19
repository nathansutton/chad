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
    """A structural `BaseEngine` that replays canned assistant turns.

    `generate` ignores the prompt and returns the NEXT string from `script`, honoring
    `CompletionEngine.generate`'s exact return contract `(text, GenStats)` so `Agent` can't
    tell it apart from a real backend. Stateless, so the warm-prefix / cache-quarantine
    members no-op (like `CompletionEngine`). If the script runs dry the loop failed to
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
    assert [m["name"] for m in agent.messages if m.get("role") == "tool"] == ["read"]


def test_agent_loop_surfaces_a_real_dispatch_failure(tmp_path, monkeypatch):
    """Negative control (verify): if a dispatch genuinely fails, the loop
    must NOT silently 'succeed'. Pointing `write` at a path under a non-existent file (so
    the parent isn't a directory) makes the real tool raise; the loop feeds the error back
    as the tool result rather than pretending the file was written. (The churn
    handoff would rightly bounce the empty-diff done first — disabled here; this test
    is about dispatch, and the handoff has its own coverage in test_done_audit.py.)"""
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

