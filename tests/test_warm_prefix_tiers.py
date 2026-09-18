"""The two-tier on-disk warm start: a fresh working directory must hit the checkpoint.

Before 2.0.3 the warm-prefix checkpoint was keyed on the whole rendered system prompt —
working directory, workspace listing and project docs included — so a session in a new
directory could never hit it and paid the full ~2.5k-token cold prefill every time
(measured: 32 of 32 fresh-directory cells in benchmarks/matrix missed, ~24 s each). Now
the engine keeps two checkpoints: the FULL prefix (same project -> zero prefill) and its
static HEAD, the tool schemas + behavioral prompt that are byte-identical in every
project (any project -> restore the head, prefill only the per-project tail).

Engine tests drive `Engine.warm_prefix` with no weights, over real mlx_lm KV caches and
real checkpoint files: a stand-in `_prefill` stores each token id in the cache as its
key, so a layer's contents name exactly the tokens resident in it. Agent tests prove
`_static_head_ids` is identical across two different directories.
"""
import json
import os
import types

import pytest

from chad import engine as E

# --------------------------------------------------------------------------- engine

def _engine(tmp_path):
    """An Engine with no weights over two real KVCache layers, checkpointing into
    `tmp_path`. `_prefill` feeds 4-token chunks honouring should_stop between chunks,
    like the real one, and writes each token id into every layer as its key."""
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    eng = object.__new__(E.Engine)
    eng.model_id = "test-model"
    eng.kv_bits = 0
    eng.cache_dir = str(tmp_path)
    eng.kv_cache_max_bytes = 0
    eng.model = types.SimpleNamespace(layers=[0, 1])
    eng._warm_prefix_ids = None
    eng._warm_head_ids = None
    eng._reset_cache()
    eng.fed = []                       # every _prefill call's ids, in order

    def _prefill(ids, should_stop=None, chunk=None, on_progress=None, on_chunk=None):
        ids = list(ids)
        done = 0
        while done < len(ids):
            if should_stop and should_stop():
                break
            piece = ids[done:done + 4]
            kv = mx.array(piece, dtype=mx.float32).reshape(1, 1, len(piece), 1)
            for layer in eng._cache:
                layer.update_and_fetch(kv, kv)
            done += len(piece)
        eng.fed.append(ids[:done])
        return done

    eng._prefill = _prefill
    return eng


def _ids(layer):
    """The token ids resident in one KV layer of an `_engine` cache."""
    if not layer.offset:
        return []
    keys, _ = layer.state
    return [int(v) for v in keys[0, 0, :, 0].tolist()]


def _ckpt_ids(path):
    """The token ids a checkpoint file holds, read back through mlx_lm's own loader."""
    return _ids(E.cache_utils.load_prompt_cache(path)[0])


HEAD = list(range(100, 112))           # tool schemas + behavioral prompt (12 tokens)
TAIL_A = [1, 2, 3, 4, 5, 6]            # project A's cwd / listing / docs
TAIL_B = [7, 8, 9, 10, 11]             # project B's


def test_cold_miss_persists_head_and_full(tmp_path):
    eng = _engine(tmp_path)
    status, n = eng.warm_prefix(HEAD + TAIL_A, head_ids=HEAD)
    assert (status, n) == ("miss", len(HEAD) + len(TAIL_A))
    assert eng.fed == [HEAD, TAIL_A]                 # head first, then the tail
    assert eng._cached_ids == HEAD + TAIL_A
    assert _ids(eng._cache[0]) == HEAD + TAIL_A
    assert eng._warm_prefix_ids == HEAD + TAIL_A and eng._warm_head_ids == HEAD
    head_path, full_path = eng._ckpt_path(HEAD), eng._ckpt_path(HEAD + TAIL_A)
    assert os.path.isfile(head_path) and os.path.isfile(full_path)
    assert _ckpt_ids(head_path) == HEAD              # the head file holds ONLY the head
    assert _ckpt_ids(full_path) == HEAD + TAIL_A


def test_fresh_directory_is_a_partial_hit(tmp_path):
    # THE fix: a second project shares nothing with the first except the head, and
    # before 2.0.3 that meant a full cold prefill. Now it restores the head and
    # prefills only its own tail.
    first = _engine(tmp_path)
    first.warm_prefix(HEAD + TAIL_A, head_ids=HEAD)

    eng = _engine(tmp_path)
    status, n = eng.warm_prefix(HEAD + TAIL_B, head_ids=HEAD)
    assert (status, n) == ("partial", len(HEAD))
    assert eng.fed == [TAIL_B]                       # only the tail was prefilled
    assert _ids(eng._cache[0]) == HEAD + TAIL_B      # head from disk + tail fed
    assert eng._cached_ids == HEAD + TAIL_B
    assert eng._warm_prefix_ids == HEAD + TAIL_B
    # and project B's full prefix is now checkpointed too: its next session is a hit
    assert os.path.isfile(eng._ckpt_path(HEAD + TAIL_B))


def test_same_project_is_a_full_hit(tmp_path):
    first = _engine(tmp_path)
    first.warm_prefix(HEAD + TAIL_A, head_ids=HEAD)
    eng = _engine(tmp_path)
    status, n = eng.warm_prefix(HEAD + TAIL_A, head_ids=HEAD)
    assert (status, n) == ("hit", len(HEAD) + len(TAIL_A))
    assert eng.fed == []                             # zero prefill
    assert _ids(eng._cache[0]) == HEAD + TAIL_A


def test_head_that_is_not_a_prefix_is_ignored(tmp_path):
    eng = _engine(tmp_path)
    status, n = eng.warm_prefix(HEAD + TAIL_A, head_ids=[999, 998])
    assert (status, n) == ("miss", len(HEAD) + len(TAIL_A))
    assert eng.fed == [HEAD + TAIL_A]                # one plain prefill, as before
    assert not os.path.isfile(eng._ckpt_path([999, 998]))
    # so is a "head" that is the whole prefix (nothing to gain from a second tier)
    eng2 = _engine(tmp_path / "fresh")
    eng2.warm_prefix(HEAD + TAIL_A, head_ids=HEAD + TAIL_A)
    assert eng2.fed == [HEAD + TAIL_A]


def test_no_head_keeps_the_old_single_tier_contract(tmp_path):
    eng = _engine(tmp_path)
    status, n = eng.warm_prefix(HEAD + TAIL_A)
    assert (status, n) == ("miss", len(HEAD) + len(TAIL_A))
    assert eng.fed == [HEAD + TAIL_A]
    eng2 = _engine(tmp_path)
    assert eng2.warm_prefix(HEAD + TAIL_A) == ("hit", len(HEAD) + len(TAIL_A))


def test_interrupted_tail_records_only_fed_tokens_and_persists_nothing(tmp_path):
    # The invariant test_interrupted_prefill_records_only_fed_tokens guards, on the new
    # partial path: _cached_ids must equal what is actually resident.
    first = _engine(tmp_path)
    first.warm_prefix(HEAD + TAIL_A, head_ids=HEAD)
    eng = _engine(tmp_path)
    checks = {"n": 0}

    def should_stop():
        checks["n"] += 1
        return checks["n"] > 1                      # one 4-token chunk, then stop

    status, n = eng.warm_prefix(HEAD + TAIL_B, head_ids=HEAD, should_stop=should_stop)
    assert (status, n) == ("partial", len(HEAD))
    assert eng._cached_ids == HEAD + TAIL_B[:4]
    assert _ids(eng._cache[0]) == HEAD + TAIL_B[:4]
    assert not os.path.isfile(eng._ckpt_path(HEAD + TAIL_B))


def test_divergence_reload_falls_back_to_the_head(tmp_path):
    # _sync_to on the non-trimmable hybrid rebuilds from the warm-prefix checkpoint on
    # a divergence; if the full one has been evicted, the head is the next best base.
    eng = _engine(tmp_path)
    eng.warm_prefix(HEAD + TAIL_A, head_ids=HEAD)
    os.remove(eng._ckpt_path(HEAD + TAIL_A))
    eng._reset_cache()
    assert eng._reload_warm_prefix(HEAD + TAIL_A + [42]) == len(HEAD)
    assert _ids(eng._cache[0]) == HEAD and eng._cached_ids == HEAD
    # a target that no longer begins with the head gets nothing
    eng._reset_cache()
    assert eng._reload_warm_prefix([5] + HEAD) == 0


def test_budget_protects_both_live_checkpoints(tmp_path):
    # A one-byte cap evicts every checkpoint the engine does not protect, so the stale
    # file going proves the cap ran, and both live files surviving proves both were
    # protected — the head too, which the full prefix's save does not write.
    eng = _engine(tmp_path)
    stale = tmp_path / "warm-stale.safetensors"
    stale.write_bytes(b"x" * 64)
    eng.kv_cache_max_bytes = 1
    eng.warm_prefix(HEAD + TAIL_A, head_ids=HEAD)
    assert not stale.exists()
    assert os.path.isfile(eng._ckpt_path(HEAD))
    assert os.path.isfile(eng._ckpt_path(HEAD + TAIL_A))


# --------------------------------------------------------------------------- agent

class _CharTok:
    """A tokenizer whose ids ARE the rendered characters, so two renders share exactly
    the token prefix their texts share — enough to exercise the differencing trick
    `_stable_prefix_ids` / `_static_head_ids` rely on, without a real chat template."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                            enable_thinking=False):
        text = "<tools>" + json.dumps(tools or [], sort_keys=True) + "</tools>"
        for m in messages:
            text += f"<{m['role']}>" + (m.get("content") or "") + f"</{m['role']}>"
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class _Engine:
    def __init__(self):
        self.model_id = "chartok-test"
        self.effective_ctx = 24000
        self.reasoning_effort_default = None
        self.cache_dir = None
        self._cached_ids = []
        self.tok = _CharTok()
        self.warm_calls = []

    def generate(self, *a, **k):
        raise AssertionError("not generating in this test")

    def reset(self):
        self._cached_ids = []

    def warm_prefix(self, prefix_ids, should_stop=None, head_ids=None):
        self.warm_calls.append((list(prefix_ids), list(head_ids or [])))
        return "skip", 0


def _agent_in(path, monkeypatch, files):
    from chad.agent import Agent
    path.mkdir(exist_ok=True)
    for name, body in files.items():
        (path / name).write_text(body)
    monkeypatch.chdir(path)
    return Agent(_Engine(), mode="yolo", thinking=False)


def test_static_head_is_identical_across_directories(tmp_path, monkeypatch):
    a = _agent_in(tmp_path / "a", monkeypatch, {"alpha.py": "x = 1\n",
                                                 "CLAUDE.md": "Use tabs.\n"})
    b = _agent_in(tmp_path / "b", monkeypatch, {"beta.py": "y = 2\n",
                                                 "gamma.py": "z = 3\n"})
    head_a, head_b = a._static_head_ids(), b._static_head_ids()
    full_a, full_b = a._stable_prefix_ids(), b._stable_prefix_ids()
    assert head_a and head_a == head_b                      # the same in both projects
    assert full_a != full_b                                 # the per-project tails differ
    assert full_a[:len(head_a)] == head_a and full_b[:len(head_b)] == head_b
    assert len(head_a) < len(full_a) and len(head_b) < len(full_b)
    # the head carries the tool schemas and the behavioral prompt, not the cwd
    head_text = a.engine.tok.decode(head_a)
    assert "<tools>" in head_text and "Working directory" not in head_text
    assert "Working directory" in a.engine.tok.decode(full_a)


def test_static_prompt_split_and_custom_prompt(tmp_path, monkeypatch):
    from chad.prompt import build_system_prompt, static_system_prompt
    agent = _agent_in(tmp_path / "p", monkeypatch, {"main.py": "pass\n"})
    agent.engine.cache_dir = str(tmp_path / "kv")     # arm the warm-start path
    assert build_system_prompt().startswith(static_system_prompt())
    assert "Working directory" not in static_system_prompt()
    # a custom system prompt (not chad's own) has no static head to checkpoint
    agent2 = _agent_in(tmp_path / "q", monkeypatch, {})
    agent2.messages[0]["content"] = "You are a different assistant."
    assert agent2._static_head_ids() == []


def test_static_head_renders_with_the_engine_effort_default(tmp_path, monkeypatch):
    """A template that opens its system block with a reasoning-effort sentence (the
    upstream Prism pack's, default xhigh) renders differently with the engine's carried
    default than without it. The static head has to be rendered the way the full prefix
    is, or it is not a prefix of it, the guard returns no head, and every fresh
    directory pays the cold prefill the head checkpoint exists to skip."""
    class _EffortTok(_CharTok):
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                                enable_thinking=False, reasoning_effort="xhigh"):
            body = super().apply_chat_template(messages, tools, add_generation_prompt,
                                               enable_thinking)
            return [ord(c) for c in f"<effort>{reasoning_effort}</effort>"] + body

    monkeypatch.delenv("CHAD_REASONING_EFFORT", raising=False)
    agent = _agent_in(tmp_path / "e", monkeypatch, {"main.py": "pass\n"})
    agent.engine.tok = _EffortTok()
    agent.engine.reasoning_effort_default = "medium"
    head, full = agent._static_head_ids(), agent._stable_prefix_ids()
    assert head and full[:len(head)] == head
    assert agent.engine.tok.decode(head).startswith("<effort>medium</effort>")
