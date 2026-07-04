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
